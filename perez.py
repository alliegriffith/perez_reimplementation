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
from util import saveModel

import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs("perez_models3", exist_ok=True)


#### differences from Perez et al. ####
# update Use a smaller critic model (hidden layers of size 1024 not 2048) bc training much smaller model (8B v 280B)
# Use batch size of 12 not 16 for memory reasons
# Use Adam not adafactor optimizers
#-------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
actor_device = torch.device("cuda:0") #  actor and critic on same device
critic_device = torch.device("cuda:0")
defender_device = torch.device("cuda:1") 
# inherently white box for now, can change later
model_name = "meta-llama/Llama-3.1-8B" 

# Training hyperparameters
NUM_STEPS = 10000 # consistent with ASTPrompter
LEARNING_RATE = 2e-6 # consistent with Perez
KL_COEF = 0.3  # KL penalty of most effective perez model
CHECKPOINT_PATH = "best_perez"
MAX_GEN_TOKENS = 24 # consistent with ASTPrompter??? ASTPrompter is 24 ### change later
BATCH_SIZE = 8 # Perez had 16, reduced for space reasons
NUM_BATCHES = NUM_STEPS // BATCH_SIZE # run same number training steps as astprompter
# Set up wandb
wandb.init(
    project="perez-training",
    name=f"perez_run_batch{BATCH_SIZE}_newMay26",
    config={
        "model_name": model_name,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "kl_coef": KL_COEF,
        "max_gen_tokens": MAX_GEN_TOKENS,
    }
)
# Load Models & Tokenizers - using Huggingface automodels which instantiate helpful classes
logging.info("Loading models and tokenizers...") # model using: "meta-llama/Llama-3.1-8B" 
# Adversary (policy) model – fine-tuned via A2C; bottom 80% of transformer layers are frozen
adv_model = AutoModelForCausalLM.from_pretrained(model_name).to(actor_device)
adv_tokenizer = AutoTokenizer.from_pretrained(model_name)
adv_tokenizer.pad_token = adv_tokenizer.eos_token
adv_tokenizer.padding_side = "left"

# Defender model (used to generate the reply, left unchanged during training)
def_model = AutoModelForCausalLM.from_pretrained(model_name).to(defender_device)
def_tokenizer = AutoTokenizer.from_pretrained(model_name)
def_tokenizer.pad_token = def_tokenizer.eos_token
def_tokenizer.padding_side = "left"

# Freeze first 50% of Adversary Layers -- follows perez et al's approach (OG did 80% but our model is much smaller)
transformer_layers = adv_model.model.layers
num_layers = len(transformer_layers)
freeze_layers = int(0.5 * num_layers)
logging.info(f"Total transformer layers = {num_layers}. Freezing first {freeze_layers} layers.")
for i, layer in enumerate(transformer_layers):
    if i < freeze_layers:
        for param in layer.parameters():
            param.requires_grad = False

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

# optimizer = adafactor -following perez
# actor_optimizer = Adafactor(
#     adv_model.parameters(),
#     lr=LEARNING_RATE,
#     relative_step=False,   # <-- manually control learning rate
#     scale_parameter=False  # optional, often paired with relative_step=False
# )
# trying adam optimizer
#actor_optimizer = torch.optim.AdamW(adv_model.parameters(), lr=LEARNING_RATE)
actor_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, adv_model.parameters()), 
    lr=LEARNING_RATE
)

# Critic def 
class Critic(nn.Module):
    def __init__(self, hidden_size=4096): # hard coded llama3.1-8b hidden size
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, 2048), 
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.value_head = nn.Linear(2048, 1) # layer norm?

    # returns the value for each token in the adversarial utterance (predicts final reward at each token)
    def forward(self, transformer_output):
        x = self.backbone(transformer_output)
        ans = self.value_head(x)
        ans = ans.squeeze(-1)
        # print("before squeeze", ans.shape) # [1, 24, 1]
        #print("critic return after squeeze", (ans.shape)) # [8, 24] # we are returning value estimate per token of adv utt
        
        return ans 


critic = Critic().to(critic_device)
# perez does not mention optimizer for critic, but we will use AdamW with 1e-4 learning rate
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4) 
# critic_optimizer = Adafactor(
#     critic.parameters(),
#     lr=1e-4,
#     scale_parameter=False,  # Needed to use the fixed LR like Perez
#     relative_step=False
# )

avg_r_best = float("-inf")
wandb.watch(adv_model, log="all", log_freq=10)
wandb.watch(critic, log="all", log_freq=10)
adv_model.train()
critic.train()


# Training time!!! 
for batch in tqdm(range(NUM_BATCHES)):
    # using autocast to enable bfloat16 training (follows perez)
    with amp.autocast("cuda", dtype=torch.bfloat16): # I think this is correct, double check? 
        # prompts is a list of randomly selected reddit prompts
        prompts = random.choices(reddit_prompts, k=BATCH_SIZE)
        prompt_inputs = adv_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_inputs = {k: v.to(actor_device) for k, v in prompt_inputs.items()}  
        prompt_ids = prompt_inputs["input_ids"]
        
        attention_mask = prompt_inputs["attention_mask"] # prompt attention mask - left padded so 0s on left
        # batch of adv outputs correspnding to the batch of prompts 
        # .gen produces tokens
        outputs = adv_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_GEN_TOKENS,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=adv_tokenizer.eos_token_id,
            output_hidden_states=True
        )
        # adversarial utterances are the generated tokens (after the prompt)
        #print(outputs.sequences.shape) # [12, 76]
        #print("outputs.sequences", outputs.sequences.shape) # [B, P + G] / [8, 52]
        gen_ids = outputs.sequences[:, prompt_ids.shape[1]:] # gen_ids is only adv utt tokens [bch, adv_utt_len]
        # print("gen_ids shape", gen_ids.shape) # [12, 24] / [B,G]
       
        # Get adversary logits (on GPU 0) - give attention mask when calc adv_logits
        # attention mask sets padding tokens in prompt and generation to 0 - expand prompt att_mask to have 1s for gen_ids
        full_attention_mask = torch.cat([attention_mask, torch.ones_like(gen_ids, device=actor_device)], dim=1)
        
        
        #### calc logits hard way bc need to condition on prompt to get KL div for gen_ids
        full_adv_output = adv_model(outputs.sequences.to(actor_device), attention_mask=full_attention_mask, output_hidden_states=True, return_dict=True) # [B, Prompt Ids + Gen Ids, vocab size]
        adv_logits = full_adv_output.logits
        adv_final_hidden = full_adv_output.hidden_states[-1] 
        #adv_logits = adv_model(full_input_ids.to(actor_device)).logits

        # Get reference logits (on GPU 1)
        with torch.no_grad():
            ref_logits = def_model(outputs.sequences.to(defender_device), attention_mask=full_attention_mask.to(defender_device)).logits
            #ref_logits = def_model(full_input_ids.to(defender_device)).logits


        # slice the prompt logits off 
        start = prompt_ids.shape[1]
        adv_logits = adv_logits[:, start:, :]
        ref_logits = ref_logits[:, start:, :]
        adv_final_hidden = adv_final_hidden[:, start:, :] 
        
        # move both logits to actor device -> KL divergence calc there
        adv_probs= F.softmax(adv_logits, dim=-1) # dim = -1 means taking softmax over the vocab size (sums to 1)
        # print("adv_probs",adv_probs.shape) # [12, G/24, 128256]
        # print("ref_logits", ref_logits.shape) # [12, G, 128256]
        # print("adv_logits", adv_logits.shape) # [12, G, 128256]
        adv_log_probs = F.log_softmax(adv_logits, dim=-1) #[12, 24, 128256] / [batch, G, vocab_size]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1).to(actor_device)
        

        # calc KL divergence - we are doing per token reward so need per token KL divergence
        ########################## HELP- PER TOKEN OR PER BATCH? ########################################
        kl_divs = (adv_probs * (adv_log_probs - ref_log_probs)).sum(dim=(2)) # sum over all vocab (2) per token in batch  -> [B, G]

        #print("kl_divs shape", kl_divs.shape) # [B, G] / [12,24]
        
        # Evaluate defender responses and compute rewards
        actor_loss = 0
        critic_loss = 0
        total_reward = 0
        total_tox_reward = 0
        total_kl_penalty = 0
        
        # gather the adv log probs of the generated tokens at those time steps
        # want log_probs = log prob (ak | sk) - gather prob at vocab dim (2) corr to gen_ids
        token_log_probs = adv_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G] 
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
                
                rewards[b, :] = tox_reward # all tokens get reward of def utt
                    
                tox_sum += tox_reward
            
        #     print(rewards)
        # print("rewards ^^ and their shape:", rewards.shape) # [8, 24] , all on e_03 - e+00 range
        
        # subtract KL penalty from rewards
        rewards = rewards.to(critic_device).detach()
        rewards = (rewards - KL_COEF * kl_divs).detach()
        #print("rewards after KL penalty", rewards.shape) # [B, G] / [8, 24]
        #print("adv_final_hidden shape", adv_final_hidden.shape) # [B, G, 4096] - hidden size of llama3.1-8b
        values = critic(adv_final_hidden.detach()) # values should be [B, G]
        #print("values shape (should be [B, G])", values.shape) # [8, 24]
        # move rewards to critic device
        #################3HELP####################
        reward_to_go = rewards.flip(dims=[1])         # [B, G] reversed in token-axis
        reward_to_go = reward_to_go.cumsum(dim=1)     # prefix sums in reversed order
        reward_to_go = reward_to_go.flip(dims=[1])    # back to original order
        # print("reward to go shape", reward_to_go.shape) # [B, G] / [8, 24]
        # print("reward to go", reward_to_go) # should be [B, G] / [8, 24] - rewards to go for each token in adv utt
        # critic loss = .5 * (V(s) - reward to go)^2 #is this correct? [B, G] -> B (rewards has no gradients, values does)
        critic_loss_batches = .5 * ((values - reward_to_go) ** 2).sum(dim=1)
        critic_loss = critic_loss_batches.mean()
        
        # update critic!
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        
        
        # add KL penalty to rewards
        values = values.detach()
        values = values.to(actor_device)
        V_next = torch.cat([values[:, 1:], torch.zeros(B, 1, device=values.device)], dim=1)
        # calc advantages - MC: reward to go - V(s)  should we do?: (TD : r + y*V(s') - V(s)) - most say yes, confused bc have reward to go
        advantages = rewards + V_next - values
        #advantages = [rewards[:, i] + values[:, i + 1] - values[:, i] for i in range(MAX_GEN_TOKENS)] # [B, G] - [B, G] -> [B, G]
       
        #  advantages and log probs should be [B, G] -> sum across G then mean across B
        # calc loss using non-normalized advantages (sum of advantages for each token in adversarial generation weighted by the token's log_prob)
        actor_loss_batches = (token_log_probs * advantages).sum(dim = 1) # sum across each token -> [B]
        actor_loss = - actor_loss_batches.mean()
        
        # update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(adv_model.parameters(), 1.0)
        actor_optimizer.step()
     
        # calc batch averages for logging
        batch_mean_advantage = advantages.mean()
        avg_reward = rewards.mean()
        avg_tox_reward = tox_sum / BATCH_SIZE
        #avg_kl_penalty = total_kl_penalty / BATCH_SIZE
        logging.info(f"[{batch}] avg_reward={avg_reward:.4f}, batch_mean_advantage={batch_mean_advantage:.4f}, avg_actor_loss={actor_loss.item():.4f}, avg_critic_loss={critic_loss.item():.4f}")
        
        wandb.log({
            "avg_reward": avg_reward,
            "batch_mean_advantage": batch_mean_advantage,
            "avg_actor_loss": actor_loss.item(),
            "avg_critic_loss": critic_loss.item(),
            "batch": batch,
            "avg_toxicity_reward": avg_tox_reward,
            #"avg_kl_penalty": avg_kl_penalty,
        })

        # dev score used by astprompter:  dev_score = sum(rewards)/len(rewards) / average rewards
        if avg_reward > avg_r_best:
            avg_r_best = avg_reward
            #torch.save(critic.state_dict(), f"perez_models/critic_best_batch{batch}.pt")
            saveModel(
                adv_model,
                adv_tokenizer,
                f"perez_model_new/adv_best",
                extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
            )    

        
        if batch % 300 == 0:
            # save the model every 100 batches
            #torch.save(critic.state_dict(), f"perez_models/critic_checkpoint_critic_batch{batch}.pt")
            saveModel(
                adv_model,
                adv_tokenizer,
                f"perez_model_new/adv_checkpoint_batch{batch}",
                extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
            )
        
        # save every number of steps just save a checkpoint 
        # end of one epoch, should loop back to start of training
        torch.cuda.empty_cache()
wandb.finish()
# Save the final model
torch.save(adv_model.state_dict(), "perez_model_new/adv_final.pt")
#torch.save(critic.state_dict(),"perez_models/critic_best.pt")