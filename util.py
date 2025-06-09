import os
import json
import torch
import torch.nn as nn
import random
import math 
from llamaguard3 import moderate

def saveModel(model, tokenizer, save_dir, extra_metadata=None):
    """
    Save a HuggingFace model (or compatible model) and tokenizer to `save_dir`.
    Also optionally saves metadata as meta.json.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights and config
    # Save model weights in sharded safetensors format with index file
    model.save_pretrained( # chatgpt helped me fix this
        save_dir,
        safe_serialization=True,              # saves .safetensors format
        max_shard_size="2GB"                  # ensure large models are saved in shards with index
    )
    
    # Save tokenizer (this adds vocab.json, merges.txt, tokenizer_config.json, etc.)
    tokenizer.save_pretrained(save_dir)

def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def gaeAndVt(rewards, values):
 
    B, T = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards).to(device)
    last_gae = 0
    gamma = 1.0 # perez and openai paper
    lam = 0.95 # openai paper

    # calc GAE in reversed order
    for t in reversed(range(T)):
        if t == T - 1: # if at last timeste, next val is 0
            next_value = 0
        else:
            next_value = values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    # print("advantages", advantages.shape)
    # print("values", values.shape)
    returns = advantages + values

    # whiten advantages (with shifted mean)
    #advantages = whiten(advantages, shift_mean=True)

    return advantages, returns

# return tokenized prompts and their attention masks
def get_prompts(reddit_prompts, adv_tokenizer, actor_device, BATCH_SIZE):
    # prompts is a list of randomly selected reddit prompts
    prompts = random.choices(reddit_prompts, k=BATCH_SIZE)
    prompt_inputs = adv_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_inputs = {k: v.to(actor_device) for k, v in prompt_inputs.items()}  
    prompt_ids = prompt_inputs["input_ids"]
    #print("prompt_ids", prompt_ids)
    attention_mask = prompt_ids != adv_tokenizer.pad_token_id
    return prompt_ids, attention_mask
    
def get_adv_utt(input_ids, input_attention_mask, model, adv_tokenizer, MAX_GEN_TOKENS):
    model.eval()
    outputs = model.base_model.generate(
        input_ids=input_ids,
        attention_mask=input_attention_mask,
        do_sample=True,         
        top_k=0,                # same as openai/huggingface re-imp paper - encourages diverse actions (not just greedy)!
        top_p=1.0,              # same as openai/huggingface re-imp paper
        temperature=1.0,
        max_new_tokens=MAX_GEN_TOKENS,
        pad_token_id=adv_tokenizer.pad_token_id,
        return_dict_in_generate=True
    )

    model.train()
    
    start = input_ids.shape[1] 
    gen_ids = outputs.sequences[:, start:]
    
    gen_mask = gen_ids != adv_tokenizer.pad_token_id
    
    return gen_ids, gen_mask, outputs

def get_defender_response(adv_tokenizer, outputs, def_tokenizer, defender_device, def_model, gen_mask, full_attention_mask, MAX_GEN_TOKENS):
    with torch.no_grad(): # no gradients needed for defender
        
        def_input_strs = adv_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True) # list of str, each str is example
    
        # def input = tensor of [B, P + G] - class from huggingface
        raw = def_tokenizer(def_input_strs, return_tensors="pt", padding=True, truncation=True)
        def_input = { k: v.to(defender_device) for k,v in raw.items() }

        def_output = def_model.generate(input_ids=def_input["input_ids"], attention_mask = def_input["attention_mask"], max_new_tokens=MAX_GEN_TOKENS,
            pad_token_id=def_tokenizer.pad_token_id, return_dict_in_generate=True)
    
        #def_output.sequences = [B, P + G + D] H = 0: [8, 75] -> H = 1: [8, 123] -> H = 2: [8, 171]
        #print(def_output.sequences.shape)
        def_input_len = def_input["input_ids"].shape[1]
        #print("def input len", def_input_len) # 52 - P + G ? 
        def_ut_tokens = def_output.sequences[:, def_input_len:]
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
            tox_reward = tox_reward # tried *10 on last run to see if improved learning signal
            #tox_reward = score
            
            #rewards[b, -1] = tox_reward # only the last token gets the reward "when extracting rewards, it is going to identify the last_response_index, the index before the EOS token (#L11-L13), and extract the reward at that index"
            last_real_token = gen_mask[b].nonzero()[-1].item()
            # print(gen_ids[b, :])
            # print(gen_ids[b, last_real_token])
            rewards[b, last_real_token] = tox_reward # setting reward to last real token in gen adv ut (sometimes shorter, not often)
            #print(rewards[b,:]) # correct [0,0,0...0.2812]
            tox_sum += tox_reward
        
        # return the whole convo
        convo_tokens = def_output.sequences
        convo_mask = def_output.sequences != adv_tokenizer.pad_token_id
        
            
    return rewards, tox_sum, convo_tokens, convo_mask

