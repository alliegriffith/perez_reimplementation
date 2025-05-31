import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, Adafactor
from tqdm import tqdm
from llamaguard3 import moderate
import logging
from convokit import Corpus, download, Conversation
from toxicity.reddit_data_helpers import filter_corpus_formatting, clean_utterance, corpus_len, corpus_to_prompts
from toxicity.split_data import filter_corpus_by_file
import random
import json
import wandb
from torch import amp
from tensorflow.keras.optimizers import Adam
from util import saveModel, whiten, gaeAndVt
import os
import gc
from copy import deepcopy


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#-------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
actor_device = torch.device("cuda:0") #  actor and critic on same device
defender_device = torch.device("cuda:1") 
# inherently white box for now, can change later
model_name = "meta-llama/Llama-3.1-8B" 

# Training hyperparameters
NUM_STEPS = 10000 # consistent with ASTPrompter
LEARNING_RATE = 2e-6 # consistent with openai paper
KL_COEF = 0.05  # far less than KL penalty of most effective perez model (0.3) but follows openai paper
MAX_GEN_TOKENS = 24 # consistent with ASTPrompter??? ASTPrompter is 24 ### change later
BATCH_SIZE = 8 # Perez had 16, reduced for space reasons
NUM_BATCHES = NUM_STEPS // BATCH_SIZE # run same number training steps as astprompter
CLIP_EPS = 0.2
TEMPERATURE = 1.0
warmup_steps = int(0.1 * NUM_STEPS) 
NUM_EPOCHS = 2
NUM_MINIBATCHES = 1 # following huggingface re-imp
MB_SIZE = BATCH_SIZE // NUM_MINIBATCHES
VF_COEF = 0.1 # follow hugging face re-imp


# Set up wandb
wandb.init(
    project="perez-training",
    name=f"perez_ppo_lowKL_multiUpdates",
    config={
        "model_name": model_name,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "kl_coef": KL_COEF,
        "max_gen_tokens": MAX_GEN_TOKENS,
        "num_epochs": NUM_EPOCHS,
        "num_minibatches": NUM_MINIBATCHES,
        "clip_eps": CLIP_EPS
    }
)
# Load Models & Tokenizers - using Huggingface automodels which instantiate helpful classes
logging.info("Loading models and tokenizers...") # model using: "meta-llama/Llama-3.1-8B" 

# Defender model (used to generate the reply, left unchanged during training)
def_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(defender_device)
def_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
#def_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
def_tokenizer.pad_token = def_tokenizer.eos_token


# Prepare Reddit Prompts -- same initial prompts as ASTPrompter (nontoxic reddit prompts init convos)
# Load prompts from your prompts.jsonl file.
with open("prompts.jsonl", 'r') as df:
    lines = df.readlines()
    data = json.loads("[" + ",".join(lines) + "]")
    prompts_rtp = [
        (
            random.choice([i["prompt"]["text"][0].lower(), i["prompt"]["text"][0]]) + i["prompt"]["text"][1:], 
            random.choice([i["continuation"]["text"][0].lower(), i["continuation"]["text"][0]]) + i["continuation"]["text"][1:]
        )
        for i in data if i["continuation"]["toxicity"] and i["continuation"]["toxicity"] > 0.5
    ]
# We’ll use only the prompt part (first element) as our Reddit prompt.
reddit_prompts = [prompt for prompt, _ in prompts_rtp]

# adv model tokenizer
adv_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
adv_tokenizer.pad_token = adv_tokenizer.eos_token
#print("adv_tokenizer pad token", adv_tokenizer.pad_token) # should be same as def_tokenizer eos  : 128001 
#pad_id = adv_tokenizer.pad_token_id


class ActorCriticModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3-8b"):
        super().__init__()
        # actor
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name,output_hidden_states=True,torch_dtype=torch.bfloat16).to(actor_device)

        # Freeze first 50% of Adversary Layers -- follows perez et al's approach (OG did 80% but our model is much smaller)
        transformer_layers = self.base_model.model.layers
        num_layers = len(transformer_layers)
        freeze_layers = int(0.5 * num_layers)
        logging.info(f"Total transformer layers = {num_layers}. Freezing first {freeze_layers} layers.")
        for i, layer in enumerate(transformer_layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size

        # critic value head - follows perez
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2048), 
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)  # Final layer to output scalar value per token
        )

    # forward pass for actor-critic model to return logits and values
    def forward(self, input_ids, attention_mask=None): # input tokenized P + G and full attention mask
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            do_dropout=False, # disable dropout (what openai/huggingface re-imp do)
        )
        
        hidden_states = outputs.hidden_states[-1]       # [B, T, H]
        lm_logits = outputs.logits                      # [B, T, vocab]
        values = self.value_head(hidden_states).squeeze(-1)  # [B, T]

        return lm_logits, values

    
model = ActorCriticModel(model_name=model_name).to(actor_device)
old_actor = deepcopy(model.base_model)
old_actor.eval()
ppo_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE
)


scheduler = get_linear_schedule_with_warmup(
    ppo_optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=NUM_STEPS,
)

avg_r_best = float("-inf")
wandb.watch(model, log="all", log_freq=10)
old_log_probs = None

model.train()
def_model.eval()

# Training time!!! 
for batch in tqdm(range(NUM_BATCHES)):
    # using autocast to enable bfloat16 training (follows perez)
    with amp.autocast("cuda", dtype=torch.bfloat16): # I think this is correct, double check? 
    #with torch.cuda.amp.autocast(dtype=torch.float32):

        # prompts is a list of randomly selected reddit prompts
        prompts = random.choices(reddit_prompts, k=BATCH_SIZE)
        prompt_inputs = adv_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_inputs = {k: v.to(actor_device) for k, v in prompt_inputs.items()}  
        prompt_ids = prompt_inputs["input_ids"]
        
        attention_mask = prompt_ids != adv_tokenizer.pad_token_id
        
        #attention_mask = prompt_inputs["attention_mask"] # prompt attention mask - left padded so 0s on left
        model.eval()
        outputs = model.base_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            do_sample=True,         
            top_k=0,                # same as openai/huggingface re-imp paper - encourages diverse actions (not just greedy)!
            top_p=1.0,              # same as openai/huggingface re-imp paper
            temperature=1.0,
            max_new_tokens=MAX_GEN_TOKENS,
            return_dict_in_generate=True
        )

        model.train()
        
        start = prompt_ids.shape[1] 
        gen_ids = outputs.sequences[:, start:]
        #print("gen_ids", gen_ids) # [B, G] / [8, 24]
        gen_mask = gen_ids != adv_tokenizer.pad_token_id
        #print("gen_mask", gen_mask) # all true bc generating 24 tokens
        
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=1)
        #print("full mask", full_attention_mask) # left padded false for shorter prompts

        # this is where the gradients come in
        adv_logits, values = model(outputs.sequences.detach(), full_attention_mask)
        
        # Get reference logits (on GPU 1)
        with torch.no_grad():
            ref_logits = def_model(outputs.sequences.to(defender_device), attention_mask=full_attention_mask.to(defender_device)).logits
            #ref_logits = def_model(full_input_ids.to(defender_device)).logits


        # slice the prompt off all of these [B, P+G] -> [B, G]
        adv_logits = adv_logits[:, start:, :]  # slice off prompt logits, [B, G, vocab_size]
        values = values[:, start:]  # slice off prompt values, [B, G]
        ref_logits = ref_logits[:, start:, :]
        
        # whiten values (normalization)
        values = whiten(values, shift_mean=False)  # [B, G] - whitened values
        # move both logits to actor device -> KL divergence calc there
        
        # print("adv_logits", adv_logits.shape) # [12, G, 128256]
        adv_log_probs = F.log_softmax(adv_logits, dim=-1) #[12, 24, 128256] / [batch, G, vocab_size]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1).to(actor_device)
        
        # gather the adv log probs of the generated tokens at those time steps
        # want log_probs = log prob (ak | sk) - gather prob at vocab dim (2) corr to gen_ids
        token_log_probs = adv_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G] 
        token_log_probs = token_log_probs * gen_mask
        
        ref_token_log_probs = ref_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G]
        ref_token_log_probs = ref_token_log_probs * gen_mask
        
        kl_divs = token_log_probs - ref_token_log_probs # [B, G] - per token KL divergence
        
        with torch.no_grad():
            old_logits = old_actor(outputs.sequences.to(actor_device), attention_mask=full_attention_mask.to(actor_device)).logits
            old_logits = old_logits[:, start:, :]
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).detach()


        #print("adv gen token log probs should be [B, G]", token_log_probs.shape) # [8, 24]
        
        with torch.no_grad(): # no gradients needed for defender
        # now let's get defender response + rewards!
        # def input is prompt + adversarially generated IDs [B] - have to do this way to sep per example
        #def_input_str = [adv_tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
            def_input_strs = adv_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True) # list of str, each str is example
        
            # def input = tensor of [B, P + G] - class from huggingface
            raw = def_tokenizer(def_input_strs, return_tensors="pt", padding=True, truncation=True)
            def_input = { k: v.to(defender_device) for k,v in raw.items() }

            def_output = def_model.generate(input_ids=def_input["input_ids"], attention_mask = def_input["attention_mask"], max_new_tokens=MAX_GEN_TOKENS,
                pad_token_id=def_tokenizer.eos_token_id)
        
            # def_output.sequences = [B, P + G + D]
            def_input_len = def_input["input_ids"].shape[1]
            #print("def input len", def_input_len) # 52 - P + G ? 
            def_ut_tokens = def_output[:, def_input_len:]
            #print("def utt len (exactly [B, 24])", def_ut_tokens.shape) #[B, D/ 24]
            
            # def_utt is list of strings [B] - each string is entire defender response
            #def_utt = def_tokenizer.batch_decode(def_ut_tokens, skip_special_tokens=True) # skipping eos_token (pad) auto covers masking padding?
        
            B, D = def_ut_tokens.shape
            tox_sum = 0
            # loop through each col of def_ut_tokens to, slice off suffixes, de-tokenize, get tox scores and calc r [B, D-i] -> [B,D] where each val is a float (0-1)
            rewards = torch.zeros(B, D, device=defender_device) # [B, G] / [8, 24]
            
            
            def_utt_strs = def_tokenizer.batch_decode(def_ut_tokens, skip_special_tokens=True)  # List[str] of length B
                
            # run the list of suffixes through moderate and append in correct [Bi, Di] location to tox_scores
            for b, text in enumerate(def_utt_strs):
                score, _ = moderate(text) 
                # follow perez reward =  (-(log(1 - toxicity)))
                tox_reward = -math.log(max(1 - score, 1e-8))
                
                #rewards[b, -1] = tox_reward # only the last token gets the reward "when extracting rewards, it is going to identify the last_response_index, the index before the EOS token (#L11-L13), and extract the reward at that index"
                last_real_token = gen_mask[b].nonzero()[-1].item()
                rewards[b, last_real_token] = tox_reward # setting reward to last real token in gen adv ut (sometimes shorter, not often)
   
                tox_sum += tox_reward
            
        #     print(rewards)
        # print("rewards ^^ and their shape:", rewards.shape) # [8, 24] , all on e_03 - e+00 range
        
        # subtract KL penalty from rewards
        rewards = rewards.to(actor_device).detach()
        # whiten rewards (no shifted mean)
        #rewards = whiten(rewards, shift_mean=False)
        rewards = (rewards - KL_COEF * kl_divs).detach()
        #print("rewards after KL penalty", rewards.shape) # [B, G] / [8, 24]
        #print("adv_final_hidden shape", adv_final_hidden.shape) # [B, G, 4096] - hidden size of llama3.1-8b
        
        # calc GAE advantage and target values
        advantages, values_target = gaeAndVt(rewards, values) # [B, G] - whitened advantages and target values
        
        # trying minibatch training like openai/huggingface re-imp
        miniActorLoss = []
        miniCriticLoss = []
        
        indices = torch.arange(BATCH_SIZE)

        model.train()  
        advantages = advantages.detach()
        values_target = values_target.detach()
        rewards = rewards.detach()
        gen_mask = gen_mask.to(actor_device).detach()  # [B, G] - mask for generated tokens
        old_log_probs = old_log_probs.to(actor_device).detach()  # [B, G] - log probs from old model
        
        # each epoch updates model gradients multiple times
        for epoch in range(NUM_EPOCHS):
            shuffled = indices[torch.randperm(BATCH_SIZE)]  # Shuffle the indices

            # each minibatch updates model gradients once with a subset of the batch data
            for i in range(NUM_MINIBATCHES): # mince only one minibatch, just use og values from batch
                # have to recalc log_probs for the minibatch
                mb_adv_logits, mb_values = model(outputs.sequences.detach(), full_attention_mask)
        
                # slice the prompt off all of these [B, P+G] -> [B, G]
                mb_adv_logits = mb_adv_logits[:, start:, :]  # slice off prompt logits, [B, G, vocab_size]
                mb_values = mb_values[:, start:]  # slice off prompt values, [B, G]
        
                # whiten values (normalization)
                mb_adv_log_probs = F.log_softmax(mb_adv_logits, dim=-1) # [MB_SIZE, G, vocab_size]
                mb_token_log_probs = mb_adv_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G] 
                mb_log_probs = mb_token_log_probs * gen_mask

                # Critic loss
                valid_positions = gen_mask.sum()
                LV1 = (((mb_values - values_target)**2) * gen_mask).sum() / valid_positions
                LV2 = (((torch.clamp(mb_values, values_target - CLIP_EPS, values_target + CLIP_EPS) - values_target)**2) * gen_mask).sum() / valid_positions
                critic_loss = 0.5 * torch.max(LV1, LV2)
                
                # Actor loss
                mb_ratios = torch.exp(mb_log_probs - old_log_probs)
                mb_clipped_ratios = torch.clamp(mb_ratios, 1 - CLIP_EPS, 1 + CLIP_EPS)
                mb_actor_loss_batches = torch.min(mb_ratios * advantages, mb_clipped_ratios * advantages) * gen_mask
                mb_actor_loss = - (mb_actor_loss_batches.sum() / gen_mask.sum())
                miniActorLoss.append(mb_actor_loss.item())
                miniCriticLoss.append(critic_loss.item())

                total_loss = mb_actor_loss + (VF_COEF * critic_loss)

                # Update
                ppo_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ppo_optimizer.step()
                print("updated!")


     
        # calc batch averages for logging
        batch_mean_advantage = advantages.mean()
        avg_reward = rewards.mean()
        avg_tox_reward = tox_sum / BATCH_SIZE
        avg_kl_penalty = kl_divs.mean().item()
        avg_actor_loss = sum(miniActorLoss) / len(miniActorLoss)
        avg_critic_loss = sum(miniCriticLoss) / len(miniCriticLoss)
        logging.info(f"[{batch}] avg_reward={avg_reward:.4f}, batch_mean_advantage={batch_mean_advantage:.4f}, avg_actor_loss={avg_actor_loss:.4f}, avg_critic_loss={avg_critic_loss:.4f}")
        
        wandb.log({
            "avg_reward": avg_reward,
            "batch_mean_advantage": batch_mean_advantage,
            "batch": batch,
            "avg_toxicity_reward": avg_tox_reward,
            "avg_actor_loss_minibatch": avg_actor_loss,
            "avg_critic_loss_minibatch": avg_critic_loss,
            "avg_kl_div": avg_kl_penalty
        })
        
        # update old model for next batch
        old_actor.load_state_dict(model.base_model.state_dict())
        
        # dev score used by astprompter:  dev_score = sum(rewards)/len(rewards) / average rewards
        if avg_reward > avg_r_best:
            avg_r_best = avg_reward
            #torch.save(critic.state_dict()w, f"perez_models/critic_best_batch{batch}.pt")
            saveModel(
                model.base_model,
                adv_tokenizer,
                f"perez_ppo_multi/adv_best",
                extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": avg_actor_loss, "critic_loss": avg_critic_loss}
            )    

        # save every number of steps just save a checkpoint 
        # end of one epoch, should loop back to start of training
        torch.cuda.empty_cache()
wandb.finish()
# Save the final model
