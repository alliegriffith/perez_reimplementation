import os
import json
import torch
import torch.nn as nn

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

    returns = advantages + values

    # whiten advantages (with shifted mean)
    #advantages = whiten(advantages, shift_mean=True)

    return advantages, returns

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


