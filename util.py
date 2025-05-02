import os
import json

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
    
    # Save any additional metadata (optional)
    if extra_metadata is not None:
        with open(os.path.join(save_dir, "meta.json"), 'w') as f:
            json.dump(extra_metadata, f, indent=4)
