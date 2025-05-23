import torch
import torch.nn as nn
import torch.nn.functional as F
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
    name=f"perez_run_batch{BATCH_SIZE}_May8",
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
        # print("before squeeze", ans.shape) # [1, 24, 1]
        # print("after squeeze", (ans.squeeze(-1)).shape) # [1, 24] # we are returning value estimate per token of adv utt
        
        return ans.squeeze(-1) # should i make just [24] / [num tokens per adv utt?]


critic = Critic().to(critic_device)
# perez does not mention optimizer for critic, but we will use AdamW with 1e-4 learning rate
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4) 
# critic_optimizer = Adafactor(
#     critic.parameters(),
#     lr=1e-4,
#     scale_parameter=False,  # Needed to use the fixed LR like Perez
#     relative_step=False
# )

adv_best = float("-inf")
wandb.watch(adv_model, log="all", log_freq=10)
wandb.watch(critic, log="all", log_freq=10)
adv_model.train()
critic.train()


# Training time!!! 
for batch in tqdm(range(NUM_BATCHES)):
    
    # gen batch of prompts and adversarial utterances and KL divergences 
    
    # using autocast to enable bfloat16 training and stochastic rounding (follows perez)
    with amp.autocast("cuda", dtype=torch.bfloat16): # allie double check
        # prompts is a list of randomly selected reddit prompts
        prompts = random.choices(reddit_prompts, k=BATCH_SIZE)
        prompt_inputs = adv_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        print("prompt inputs:", prompt_inputs)
        prompt_inputs = {k: v.to(actor_device) for k, v in prompt_inputs.items()}  
        prompt_ids = prompt_inputs["input_ids"]
        
        attention_mask = prompt_inputs["attention_mask"] # prompt attention mask
        print("prompt attention mask", attention_mask)
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
            output_hidden_states=True, 
            return_dict_in_generate=True

        )
        # adversarial utterances are the generated tokens (after the prompt)
        #print(outputs.sequences.shape) # [12, 76]
        print("outputs.sequences" outputs.sequences)
        gen_ids = outputs.sequences[:, prompt_ids.shape[1]:] # gen_ids is only adv utt tokens [bch, adv_utt_len]
        #print("gen_ids shape", gen_ids.shape) # [12, 50]
       
        # Get adversary logits (on GPU 0) - give attention mask when calc adv_logits
        # attention mask sets padding tokens in prompt and generation to 0 - expand prompt att_mask to have 1s for gen_ids
        full_attention_mask = torch.cat([attention_mask, torch.ones_like(gen_ids, device=actor_device)], dim=1)
        # start of adv_models's gradients -> does this have gradients attached?
        
        #### input = torch cat prompt and outputs, dim = -1 -> calc logits with model()
        adv_logits = adv_model(outputs.sequences.to(actor_device), attention_mask=full_attention_mask).logits # [B, Prompt Ids + Gen Ids, vocab size]
        
        #adv_logits = adv_model(full_input_ids.to(actor_device)).logits

        # Get reference logits (on GPU 1)
        with torch.no_grad():
            ref_logits = def_model(outputs.sequences.to(defender_device), attention_mask=full_attention_mask.to(defender_device)).logits
            #ref_logits = def_model(full_input_ids.to(defender_device)).logits


        # slice the prompt logits off 
        start = prompt_ids.shape[1]
        adv_logits = adv_logits[:, start:, :]
        ref_logits = ref_logits[:, start:, :]
        
        # move both logits to actor device -> KL divergence calc there
        adv_probs= F.softmax(adv_logits, dim=-1) # dim = -1 means taking softmax over the vocab size (sums to 1)
        # print("adv_probs_all",adv_probs_all.shape) # [12, seq_len (around 80), 128256]
        # print("ref_logits", ref_logits.shape) # [12, seq_len (around 80), 128256]
        # print("adv_logits", adv_logits.shape) # [12, seq_len (around 80), 128256]
        adv_log_probs = F.log_softmax(adv_logits, dim=-1) #[12, 77, 128256] / [batch, seq_len, vocab_size]
        ref_log_probs = F.log_softax(ref_logits, dim=-1).to(actor_device)
        #ref_probs_all = F.softmax(ref_logits, dim=-1).to(actor_device) 
        
        #print("KL shape", adv_probs_all[:, start:, :].shape) #[8, 24, 128256]

        # calc KL divergence - removed the "start:-1" bc looked like wrong shape - remove the logits for the prompts ids, KL div over gen token prob distribs
        ########################## HELP- PER TOKEN OR PER BATCH? ########################################
        kl_divs = (adv_probs * (adv_log_probs - ref_log_probs)).sum(dim=(1)) # sum over all vocab (2) per token in batch  -> [B, G]

        print("kl_divs shape", kl_divs.shape) # [B]
        
        factory_kl = torch.distributions.kl_divergence(adv_probs, ref_logits) 
        
        print(f"my KL div: {kl_divs} factory kl div: {factory_kl}")
        # Evaluate defender responses and compute rewards
        actor_loss = 0
        critic_loss = 0
        total_reward = 0
        total_tox_reward = 0
        total_kl_penalty = 0
        
        # gather the adv log probs of the generated tokens at those time steps
        # want log_probs = log prob (ak | sk) - gather prob at vocab dim (2) corr to gen_ids
        token_log_probs = adv_log_probs.gather(gen_ids.unsqueeze(1), dim = 2) # [B, G] 
        print("adv gen token log probs should be [B, G]", token_log_probs.shape)
        
        batch_advantages = []
        log_probs_batch = []
        batch_rewards = []
        
        # now let's get defender response + rewards!
        # def input is prompt + adversarially generated IDs [B] - have to do this way to sep per example
        #def_input_str = [adv_tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs.sequences]
        def_input_strs = adv_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True) # list of str, each str is example
        
        # def input = tensor of [B, P + G] - class from huggingface
        def_input = def_tokenizer(def_input_strs, return_tensors="pt", padding=True, truncation=True).to(defender_device)
        
        def_output = def_model.generate(input_ids=def_input["input_ids"], attention_mask = def_input["attention_mask"], max_new_tokens=MAX_GEN_TOKENS,
                pad_token_id=def_tokenizer.eos_token_id)
        
        # def_output.sequences = [B, P + G + D]
        def_input_len = def_input["input_ids"].shape[1]
        print("def input len (about 75)", def_input_len)
        def_ut_tokens = def_output.sequences[:, def_input_len:]
        print("def utt len (exactly [B, 24])", def_ut_tokens.shape)
        
        # def_utt is list of strings [B] - each string is entire defender response
        #def_utt = def_tokenizer.batch_decode(def_ut_tokens, skip_special_tokens=True) # skipping eos_token (pad) auto covers masking padding?
    
        B, D = def_ut_tokens.shape
        tox_sum = 0
        with torch.no_grad():
            # loop through each col of def_ut_tokens to, slice off suffixes, de-tokenize, get tox scores and calc r [B, D-i] -> [B,D] where each val is a float (0-1)
            rewards = torch.zeros(B, D, device="defender_device")
            
            for i in range(D):
                suffix_token_ids = def_ut_tokens[: , i:] # want def_to_score = [B, D (num tokens defender generated)] where Bi, Di is from current token to end token in def_ut
                
                suffix_texts = def_tokenizer.batch_decode(suffix_token_ids, skip_special_tokens=True)  # List[str] of length B
                
                # run the list of suffixes through moderate and append in correct [Bi, Di] location to tox_scores
                for b, text in enumerate(suffix_texts):
                    score, _ = moderate(text) 
                    # follow perez reward =  (-(log(1 - toxicity))) - KL_COEF * kl_div
                    tox_reward = -math.log(max(1 - score, 1e-8))
                
                    rewards[b, i] = tox_reward ######### TODO : add KL penalty 
                    
                    tox_sum += tox_reward
            
            print(rewards)
        print("rewards ^^ and their shape:", rewards.shape)
        
        # now pass the final transformer representation of adv utterance through critic to get V(s) estimates -- [B, G (num adv tokens)]
        final_trans_rep = outputs.hidden_states[-1][:, start:, :].detach() #remove gradients from adv_model, this is to update critic
        
        values = critic(final_trans_rep) # values should be [B, G]
        print("values shape (should be [B, G])", values.shape)
        
        # critic loss = .5 * (V(s) - reward to go)^2 #is this correct? [B, G] -> B (rewards has no gradients, values does)
        critic_loss_batches = .5 * ((values - rewards) ** 2).sum(dim=1)
        critic_loss = critic_loss_batches.mean()
        
        # update critic!
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        
        
        B, G = gen_ids.shape
        # calc advantages - MC: reward to go - V(s)  should we do?: (TD : r + y*V(s') - V(s)) - most say yes, confused bc have reward to go
        advantages = rewards - values
       
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