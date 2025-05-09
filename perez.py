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
os.makedirs("perez_models2", exist_ok=True)


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
# Load Models & Tokenizers
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

# Freeze first 80% of Adversary Layers -- follows perez et al's approach
transformer_layers = adv_model.model.layers
num_layers = len(transformer_layers)
freeze_layers = int(0.5 * num_layers)
logging.info(f"Total transformer layers = {num_layers}. Freezing first {freeze_layers} layers.")
for i, layer in enumerate(transformer_layers):
    if i < freeze_layers:
        for param in layer.parameters():
            param.requires_grad = False
            
# also freeze the embedding layer??
# for param in adv_model.model.embed_tokens.parameters():
#     param.requires_grad = False



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



# follow perez's use of popart for the value head
# PopArt class wraps the critic's value head with PopArt normalization
# class PopArt(nn.Module):
#     # enables adaptively normalizing targets used in training
#     # step size 1e-4 consistent with perez et al
#     def __init__(self, input_size, output_size=1, beta=1e-4):
#         super().__init__()
#         self.linear = nn.Linear(input_size, output_size)
#         self.register_buffer("mean", torch.zeros(1))  
#         self.register_buffer("var", torch.ones(1))
#         self.beta = beta
#         self.epsilon = 1e-5

#     def update_stats(self, targets):
#         with torch.no_grad():
#             old_mean = self.mean.clone()
#             old_std = self.var.sqrt().clone()

#             batch_mean = targets.mean()
#             batch_var = targets.var(unbiased=False)

#             # Update the running mean and variance
#             self.mean = (1 - self.beta) * self.mean + self.beta * batch_mean.to(critic_device)
#             self.var = (1 - self.beta) * self.var + self.beta * batch_var.to(critic_device)
#             new_std = self.var.sqrt()

#             # Rescale weights and bias
#             self.linear.weight.data = self.linear.weight.data * (old_std / new_std)
#             self.linear.bias.data = (old_std / new_std) * self.linear.bias.data + (old_mean - self.mean) / new_std

#     def normalize(self, targets):
#         return (targets - self.mean) / (self.var.sqrt() + self.epsilon)

#     def forward(self, x):
#         return self.linear(x)


# Critic def 
class Critic(nn.Module):
    def __init__(self, hidden_size=4096): # hard coded llama3.1-8b hidden size
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size, 1024), # smaller hidden layers than perez bc we are training a much smaller adversary
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        # apply PopArt normalization to the value head - ensures value prediction remains stable and normalized
        #self.value_head = PopArt(1024)
        # trying a simple value head (not popart) 
        self.value_head = nn.Linear(1024, 1) # layer norm instead of popart

    # returns the value for each token in the adversarial utterance (predicts final reward at each token)
    def forward(self, transformer_output):
        x = self.backbone(transformer_output)
        ans = self.value_head(x)
        # print("before squeeze", ans.shape) # [1, 50, 1]
        # print("after squeeze", (ans.squeeze(-1)).shape) # [1, 50] # we are returning value estimate per token of adv utt
        return ans.squeeze(-1)


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
        prompt_inputs = {k: v.to(actor_device) for k, v in prompt_inputs.items()}  
        prompt_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs["attention_mask"] # prompt attention mask
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
            pad_token_id=adv_tokenizer.eos_token_id
        )
        # adversarial utterances are the generated tokens (after the prompt)
        #print(outputs.sequences.shape) # [12, 76]
        gen_ids = outputs.sequences[:, prompt_ids.shape[1]:] # gen_ids is only adv utt tokens [bch, adv_utt_len]
        #print("gen_ids shape", gen_ids.shape) # [12, 50]
       
        # Get adversary logits (on GPU 0) - give attention mask when calc adv_logits
        # attention mask sets padding tokens in prompt and generation to 0
        full_attention_mask = torch.cat([attention_mask, torch.ones_like(gen_ids, device=actor_device)], dim=1)
        # start of adv_models's gradients -> does this have gradients attached?
        adv_logits = adv_model(outputs.sequences.to(actor_device), attention_mask=full_attention_mask).logits

        #adv_logits = adv_model(full_input_ids.to(actor_device)).logits

        # Get reference logits (on GPU 1)
        with torch.no_grad():
            ref_logits = def_model(outputs.sequences.to(defender_device), attention_mask=full_attention_mask.to(defender_device)).logits
            #ref_logits = def_model(full_input_ids.to(defender_device)).logits

        # move both logits to the critic device -> KL divergence calc there
        adv_probs_all = F.softmax(adv_logits, dim=-1)
        # print("adv_probs_all",adv_probs_all.shape) # [12, seq_len (around 80), 128256]
        # print("ref_logits", ref_logits.shape) # [12, seq_len (around 80), 128256]
        # print("adv_logits", adv_logits.shape) # [12, seq_len (around 80), 128256]
        adv_log_probs_all = F.log_softmax(adv_logits, dim=-1) #[12, 77, 128256] / [batch, seq_len, vocab_size]
        #ref_probs_all = F.softmax(ref_logits, dim=-1).to(actor_device) 
        start = prompt_ids.shape[1]
        #print("KL shape", adv_probs_all[:, start:, :].shape) #[8, 24, 128256]

        # calc KL divergence - removed the "start:-1" bc looked like wrong shape
        kl_divs = (adv_probs_all[:, start:, :] * (F.log_softmax(adv_logits, dim =-1)[:, start:, :] - 
                                                    F.log_softmax(ref_logits, dim=-1).to(actor_device)[:, start:, :])).sum(dim=(1, 2))


        # Evaluate defender responses and compute rewards
        actor_loss = 0
        critic_loss = 0
        total_reward = 0
        total_tox_reward = 0
        total_kl_penalty = 0
        
        prompt_lens = attention_mask.sum(dim=1)  # shape: (batch_size,)
        # if prompt_lens != prompt_ids.shape[1]:
        #     print("Prompt length mismatch:", prompt_lens, prompt_ids.shape[1])
        #print(adv_log_probs_all.shape) #[12, 77, 128256] / [batch, seq_len, vocab_size]
        
        #print(gen_ids.shape) # [12,50] [batch, gen_len]
        # allie change this to a gather:
        # log_probs = torch.stack([
        #     torch.stack([
        #         adv_log_probs_all[i, prompt_lens[i] + j - 1, token_id]
        #         for j, token_id in enumerate(gen_ids[i])
        #     ])
        #     for i in range(BATCH_SIZE)
        # ])

        gen_pos = prompt_lens.unsqueeze(1) + torch.arange(gen_ids.shape[1], device=actor_device) - 1
        #print("gen_pos", gen_pos.shape)# Shape: [batch_size, gen_len] / [12, 50]

        # gather the log probs of the generated tokens at those time steps
        log_probs = adv_log_probs_all[
            torch.arange(BATCH_SIZE).unsqueeze(1),  
            gen_pos,                                
            gen_ids                                
        ]  # Final shape: [batch_size, gen_len] / [12, 50]

        # print("log_probs shape", log_probs.shape) #  [12, 50] /[batch size, gen_len]
        batch_advantages = []
        log_probs_batch = []
        batch_rewards = []
        
        # loop to get defender responses and compute rewards
        
        for i in range(BATCH_SIZE):
            # Get the prompt and adversarial utterance
            prompt = prompts[i]
            adv_utt = adv_tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            dialogue = prompt + " " + adv_utt
            # prepare the dialogue for the defender model
            def_input = def_tokenizer(dialogue, return_tensors="pt", padding=True, truncation=True)
            def_input = {k: v.to(defender_device) for k, v in def_input.items()}

            def_out = def_model.generate(
                input_ids=def_input["input_ids"],
                attention_mask=def_input["attention_mask"],
                max_new_tokens=MAX_GEN_TOKENS,
                pad_token_id=def_tokenizer.eos_token_id
            )

            input_len = def_input["input_ids"].shape[1]
            def_utt = def_tokenizer.decode(def_out[0][input_len:], skip_special_tokens=True)

            # use llamaguard to get defender toxicity score
            toxicity, _ = moderate(def_utt)
            #print(toxicity)
            # follow perez reward =  -(log(1 - toxicity)) - KL penalty
            #reward = - (math.log(max(1 - toxicity, 1e-8))) - KL_COEF * kl_divs[i].item()
            ################################HELP############################ 
            ## reward might be too small/weak!
            toxicity_reward =  -math.log(max(1 - toxicity, 1e-8))   # High toxicity -> big reward
            kl_penalty = KL_COEF * kl_divs[i].item()
            # OG perez reward (neg bc Adam minimizes and we want to maximize toxicity)
            reward = toxicity_reward - kl_penalty
            batch_rewards.append(reward)
            total_reward += reward
            
            # update total rewards to track average over batch
            total_tox_reward += toxicity_reward
            total_kl_penalty += kl_penalty

            ###### remove no grad if combine to one update
            with torch.no_grad():
                sequences_cuda = outputs.sequences.to(actor_device)
                ### pass critic state and adv utterance?
                critic_inputs = {
                        "input_ids": sequences_cuda[i].unsqueeze(0),
                        "attention_mask": full_attention_mask[i].unsqueeze(0).to(actor_device)
                } # MUST BE SAME SHAPE AS log_probs[i]
                # 
                new_outputs = adv_model(**critic_inputs, output_hidden_states=True, return_dict=True)
                # critic predicts value for each token in the adversarial utterance
                hidden_states = new_outputs.hidden_states[-1]  # (batch, seq_len, hidden)
                
            # values is a tensor of shape (seq_len) - estimate for each token in the adversarial utterance
            # values = critic(hidden_states.to(critic_device)).squeeze(-1) # removed to critic device bc actor/critic device is same now
            values = critic(hidden_states)  
            #print("values shape pre squeeze", values.shape) # now [1,80] bc critic gets prompt + adv utt
            values = values.squeeze(-1)
            gen_len = gen_ids.shape[1]  # the true length of the generated tokens
            values = values[:, -gen_len:]  # keep only the final gen_len tokens

            #print("values shape post squeeze", values.shape) # [1,50]


            # advantage is the true reward minus the predicted (critic) value
            # advantage is per token in the adversarial utterance
            trueReward = torch.full_like(values, fill_value=reward)
            # Advantage = R - V (no gradient)
            advantages = (trueReward - values.detach()).squeeze(0) 
            #print("advantages shape", advantages.shape) # [50]
            
            # Ensure advantage and log_probs[i] match (they should)
            assert log_probs[i].shape == advantages.shape, f"Mismatch at batch {batch}, sample {i}, log_probs shape {log_probs[i].shape}, advantages shape {advantages.shape}"    
           
            batch_advantages.append(advantages)  
            log_probs_batch.append(log_probs[i])
            
            # calc critic loss as diff beteen target (critic vals) and true rewards (defender tox)
            # critic loss is a single value - average of the MSE loss over all tokens - use popart
            # normalized_rewards = critic.value_head.normalize(trueReward).detach() # already kind of normalized
            # normalized_values = critic.value_head.normalize(values) 

            # not positive if correct to normalize values 
            critic_loss += F.mse_loss(values, trueReward.detach())
        
        # calc loss using non-normalized advantages (sum of advantages for each token in adversarial generation weighted by the token's log_prob)
        actor_loss = sum(-(adv * lp).sum() for adv, lp in zip(batch_advantages, log_probs_batch))
        
        # update popart running stats on batch of true rewards 
        # all_true_rewards = torch.stack([torch.full_like(values, fill_value=r) for r, values in zip(batch_rewards, critic_values)])
        # critic.value_head.update_stats(all_true_rewards)


        # actor loss updates the non-frozen transformer layers
        actor_loss.backward()
        # critic loss updates the critic mlp weights
        critic_loss.backward() 


        # Gradient clipping - L2 norm gradient clipping of 1 (from perez)
        torch.nn.utils.clip_grad_norm_(adv_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)

        actor_optimizer.step()
        critic_optimizer.step()
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        # calc batch averages for logging
        batch_mean_advantage = torch.stack([adv.mean() for adv in batch_advantages]).mean().item()
        avg_reward = total_reward / BATCH_SIZE
        avg_tox_reward = total_tox_reward / BATCH_SIZE
        avg_kl_penalty = total_kl_penalty / BATCH_SIZE
        logging.info(f"[{batch}] avg_reward={avg_reward:.4f}, batch_mean_advantage={batch_mean_advantage:.4f}, avg_actor_loss={actor_loss.item():.4f}, avg_critic_loss={critic_loss.item():.4f}")
        
        wandb.log({
            "avg_reward": avg_reward,
            "batch_mean_advantage": batch_mean_advantage,
            "avg_actor_loss": actor_loss.item(),
            "avg_critic_loss": critic_loss.item(),
            "batch": batch,
            "avg_toxicity_reward": avg_tox_reward,
            "avg_kl_penalty": avg_kl_penalty,
        })

        # dev score used by astprompter:  dev_score = sum(rewards)/len(rewards)
        if batch_mean_advantage > adv_best:
            adv_best = batch_mean_advantage
            #torch.save(critic.state_dict(), f"perez_models/critic_best_batch{batch}.pt")
            saveModel(
                adv_model,
                adv_tokenizer,
                f"perez_models2/adv_best",
                extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
            )

        
        if batch % 300 == 0:
            # save the model every 100 batches
            #torch.save(critic.state_dict(), f"perez_models/critic_checkpoint_critic_batch{batch}.pt")
            saveModel(
                adv_model,
                adv_tokenizer,
                f"perez_models2/adv_checkpoint_batch{batch}",
                extra_metadata={"avg_reward": avg_reward, "batch": batch, "batch_mean_advantage": batch_mean_advantage, "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
            )
        
        # save every number of steps just save a checkpoint 
        # end of one epoch, should loop back to start of training
        torch.cuda.empty_cache()
wandb.finish()
# Save the final model
torch.save(adv_model.state_dict(), "perez_models2/adv_final.pt")
#torch.save(critic.state_dict(),"perez_models/critic_best.pt")