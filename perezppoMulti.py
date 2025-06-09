import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
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
from tensorflow.keras.optimizers import Adam # try this adam optimizer if still have trouble
from util import saveModel, whiten, gaeAndVt, get_prompts, get_adv_utt, get_defender_response
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
KL_COEF = 0.01  # far less than KL penalty of most effective perez model (0.3) but follows openai paper
MAX_GEN_TOKENS = 24 # consistent with ASTPrompter??? ASTPrompter is 24 ### change later
BATCH_SIZE = 8 # Perez had 16, reduced for space reasons
NUM_BATCHES = NUM_STEPS // BATCH_SIZE # run same number training steps as astprompter
CLIP_EPS = 0.2
TEMPERATURE = 0.7 # astprompter
warmup_steps = int(0.1 * NUM_STEPS) 
NUM_EPOCHS = 4
NUM_MINIBATCHES = 1 # following huggingface re-imp
MB_SIZE = BATCH_SIZE // NUM_MINIBATCHES
VF_COEF = 0.1 # follow hugging face re-imp
# top_p = 0.7


# Set up wandb
wandb.init(
    project="perez-training",
    name=f"perez_run_ppo_multiTurn",
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
def_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
def_model.resize_token_embeddings(len(def_tokenizer)) # resize tokenizer because added new padding token
#def_tokenizer.pad_token = def_tokenizer.eos_token


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
            else:
                param.requires_grad = True

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
        # check if hidden_states is correct
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            temperature = TEMPERATURE,
            return_dict=True,
            do_dropout=False, # disable dropout (what openai/huggingface re-imp do)
        )
        
        hidden_states = outputs.hidden_states[-1]       # [B, T, H]
        lm_logits = outputs.logits / TEMPERATURE                      # [B, T, vocab]
        values = self.value_head(hidden_states).squeeze(-1)  # [B, T]

        return lm_logits, values


# code from https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
# adaptively update KL coef based on 
class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl, n_steps=1):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        multiplier = 1 + proportional_error * n_steps / self.horizon
        self.value *= multiplier
   
model = ActorCriticModel(model_name=model_name).to(actor_device)

ppo_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    eps=1e-7  
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    ppo_optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup_steps)
)
# scheduler = get_linear_schedule_with_warmup(
#     ppo_optimizer,
#     num_warmup_steps=warmup_steps,
#     num_training_steps=NUM_STEPS,
# )

# from https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo - last experiment
kl_controller = AdaptiveKLController(
    init_kl_coef=0.01,  
    target=0.2,        
    horizon=10000       
)


# adv model tokenizer
adv_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
adv_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.base_model.resize_token_embeddings(len(adv_tokenizer))

old_actor = deepcopy(model.base_model)
old_actor.eval()

print("adv_tokenizer pad token", adv_tokenizer.pad_token_id) 
print("adv_tokenizer eos token", adv_tokenizer.eos_token_id) # should be same as def_tokenizer eos  : 128001 


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
        prompt_ids, attention_mask = get_prompts(reddit_prompts, adv_tokenizer, actor_device, BATCH_SIZE)
        
        for i in range(3):
            print("horizon", i)
            if i==0:
                input_ids = prompt_ids
                input_attention_mask = attention_mask
                
            else:
                lengths = convo_mask.sum(dim=1)           # shape [B]

                max_len = lengths.max()
                B, _    = convo_tokens.size()
                new_ids  = torch.full((B, max_len), adv_tokenizer.pad_token_id, device=actor_device)
                new_mask = torch.zeros_like(new_ids)

                for i, L in enumerate(lengths):
                    new_ids[i, max_len - L : ]  = convo_tokens[i, :L]
                    new_mask[i, max_len - L : ] = 1

                input_ids = new_ids
                #print("left padded new_ids:", new_ids)
                input_attention_mask = new_mask
                #print("new attention mask:", input_attention_mask)

            
            gen_ids, gen_mask, outputs = get_adv_utt(input_ids, input_attention_mask, model, adv_tokenizer, MAX_GEN_TOKENS)
        
            full_attention_mask = torch.cat([input_attention_mask, gen_mask], dim=1)
            #print("full mask", full_attention_mask) # left padded false for shorter prompts
        
            adv_logits, values = model(outputs.sequences, full_attention_mask)
        
            with torch.no_grad():
                ref_logits = def_model(outputs.sequences.to(defender_device), attention_mask=full_attention_mask.to(defender_device)).logits
                ref_logits = ref_logits /   TEMPERATURE

            start = input_ids.shape[1] # we only want KL div of the most recent action (adv utt)
            # slice the prompt off all of these [B, P+G] -> [B, G]
            adv_logits = adv_logits[:, start:, :]  # slice off prompt logits, [B, G, vocab_size]
            values = values[:, start:]  # slice off prompt values, [B, G]
            ref_logits = ref_logits[:, start:, :]
            
            # whiten values (normalization)
            #values = whiten(values, shift_mean=False)  # [B, G] - whitened values
            adv_probs= F.softmax(adv_logits, dim=-1)
            adv_log_probs = F.log_softmax(adv_logits, dim=-1) #[12, 24, 128256] / [batch, G, vocab_size]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1).to(actor_device)
            
            # gather the adv log probs of the generated tokens at those time steps
            # want log_probs = log prob (ak | sk) - gather prob at vocab dim (2) corr to gen_ids
            token_log_probs = adv_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G] 
            token_log_probs = token_log_probs * gen_mask
            
            ref_token_log_probs = ref_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1) # [B, G]
            ref_token_log_probs = ref_token_log_probs * gen_mask
        
            #kl_divs = token_log_probs - ref_token_log_probs # [B, G] - per token KL divergence
            kl_divs = (adv_probs * (adv_log_probs - ref_log_probs)).sum(dim=(2))
        
            with torch.no_grad():
                old_logits = old_actor(outputs.sequences.to(actor_device), attention_mask=full_attention_mask.to(actor_device)).logits
                old_logits = old_logits[:, start:, :]
                old_log_probs = F.log_softmax(old_logits, dim=-1)
                old_log_probs = old_log_probs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).detach()
                old_log_probs = old_log_probs * gen_mask

            rewards, tox_sum, convo_tokens, convo_mask = get_defender_response(adv_tokenizer, outputs, def_tokenizer, defender_device, def_model, gen_mask, full_attention_mask, MAX_GEN_TOKENS)
        #     print(rewards)
        # print("rewards ^^ and their shape:", rewards.shape) # [8, 24] , all on e_03 - e+00 range
        
        # subtract KL penalty from rewards
            rewards = rewards.to(actor_device).detach()
            # whiten rewards (no shifted mean)
            #rewards = whiten(rewards, shift_mean=False)
            #rewards = (rewards - KL_COEF * kl_divs).detach()
            rewards = (rewards - kl_controller.value * kl_divs).detach()

        #print("rewards", rewards)
        #print("rewards after KL penalty", rewards.shape) # [B, G] / [8, 24]
        #print("adv_final_hidden shape", adv_final_hidden.shape) # [B, G, 4096] - hidden size of llama3.1-8b
            mask = gen_mask.float()
        # calc GAE advantage and target values
            advantages, values_target = gaeAndVt(rewards, values) # [B, G] - whitened advantages and target values
        # print("advantages shape", advantages.shape) # [B, G] / [8, 24]
        # print("values_target shape", values_target.shape) # [B, G] / [8, 24]
        # values_target is 
        #print("vals target", values_target)
            advantages = advantages * gen_mask
        #print("adv:",advantages)
            returns = values_target * gen_mask

        
            ## calc clipped surrogate loss for actor and critic
            LV1 = ((values - values_target)**2) * mask  # [B, G]
            LV2 = ((torch.clamp(values, values_target - CLIP_EPS, values_target + CLIP_EPS) - values_target)**2) * mask # [B, G]
            critic_loss = 0.5 * torch.max(LV1, LV2).sum() / mask.sum()

            # clipped PPO objective
            ratios = torch.exp(token_log_probs - old_log_probs)  # [B]
            clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS)  # typical clip_eps = 0.2

            actor_loss_batches = torch.min(ratios * advantages, clipped_ratios * advantages) * gen_mask
            actor_loss = - (actor_loss_batches.sum() / gen_mask.sum())
            #print(actor_loss)
            
            total_loss = actor_loss + (VF_COEF * critic_loss)
            
            # update old model for next batch
            old_actor.load_state_dict(model.base_model.state_dict())
            
            # update the input for the next step to be the conversation so far
            input_ids = convo_tokens.to(actor_device)
            input_attention_mask = convo_mask.to(actor_device)
            
            # update actor
            ppo_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ppo_optimizer.step()
            scheduler.step()
        
     
            # calc batch averages for logging
            batch_mean_advantage = advantages.sum() / gen_mask.sum()
            avg_reward = rewards.sum() / gen_mask.sum()
            avg_tox_reward = tox_sum / BATCH_SIZE
            avg_kl_penalty = kl_divs.mean().item()
            kl_controller.update(avg_kl_penalty, n_steps=1)
            kl_per_token = kl_divs.sum() / gen_mask.sum()
            kl_per_sequence = kl_divs.sum(dim=1).mean()


            
            logging.info(f"[{batch}] avg_reward={avg_reward:.4f}, batch_mean_advantage={batch_mean_advantage:.4f}, avg_actor_loss={actor_loss:.4f}, avg_critic_loss={critic_loss.item():.4f}")

            wandb.log({
                "avg_reward": avg_reward,
                "batch_mean_advantage": batch_mean_advantage,
                "avg_actor_loss": actor_loss.item(),
                "avg_critic_loss": critic_loss.item(),
                "batch": batch,
                "avg_toxicity_reward": avg_tox_reward,
                "avg_kl_penalty": avg_kl_penalty,
                "kl_coef": kl_controller.value,
                "kl_per_token": kl_per_token.item(),
                "kl_per_sequence": kl_per_sequence.item()
            })
        
        
            # dev score used by astprompter:  dev_score = sum(rewards)/len(rewards) / average rewards
            if avg_reward > avg_r_best:
                avg_r_best = avg_reward
                #torch.save(critic.state_dict()w, f"perez_models/critic_best_batch{batch}.pt")
                saveModel(
                    model.base_model,
                    adv_tokenizer,
                    f"perez_ppo_multi/adv_best",
                    extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
                )    
        
        
        # end of one epoch, should loop back to start of training
        #torch.cuda.empty_cache()
wandb.finish()
# Save the final model
