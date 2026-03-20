"""
Precompute GPT-2 unconditional token probabilities P(token) for all tokens in Gemma vocabulary.
This is used for typicality correction in eval.py.

For each token in Gemma's vocabulary:
1. Decode the token to text
2. Compute P_gpt2(text) using GPT-2
3. Store at the Gemma token's index

The output is saved as a numpy array where index corresponds to Gemma token_id.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from pathlib import Path
import argparse
from tqdm import tqdm


def compute_token_probability_gpt2(text, gpt2_model, gpt2_tokenizer, device):
    """
    Compute GPT-2's unconditional probability P(text).
    
    For single-token text in GPT-2, this is straightforward.
    For multi-token text in GPT-2, we compute the product of conditional probabilities.
    """
    # Tokenize without special tokens
    input_ids = gpt2_tokenizer.encode(text, add_special_tokens=False)
    
    if len(input_ids) == 0:
        return float('-inf')
    
    # For single token, compute P(token)
    if len(input_ids) == 1:
        context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)
        full_ids = context_ids + input_ids
        
        input_tensor = torch.tensor([full_ids]).to(device)
        with torch.no_grad():
            outputs = gpt2_model(input_tensor)
            logits = outputs.logits
            
            target_logits = logits[0, len(context_ids) - 1, :]
            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_prob = log_probs[input_ids[0]].item()
            
        return token_log_prob
    else:
        # For multi-token, compute product of conditional probabilities
        log_prob_sum = 0.0
        
        for i in range(len(input_ids)):
            if i == 0:
                context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)
            else:
                context_ids = gpt2_tokenizer.encode("", add_special_tokens=True)[:-1] + input_ids[:i]
            
            full_ids = context_ids + [input_ids[i]]
            input_tensor = torch.tensor([full_ids]).to(device)
            
            with torch.no_grad():
                outputs = gpt2_model(input_tensor)
                logits = outputs.logits
                
                target_logits = logits[0, len(context_ids) - 1, :]
                log_probs = torch.log_softmax(target_logits, dim=-1)
                token_log_prob = log_probs[input_ids[i]].item()
                
                log_prob_sum += token_log_prob
        
        return log_prob_sum


def compute_vocab_probabilities(gemma_model='google/gemma-2-2b', gpt2_model='gpt2', device='cuda'):
    """
    Compute P_gpt2(token) for all tokens in Gemma's vocabulary.
    """
    print(f"Loading Gemma tokenizer from {gemma_model}...")
    gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model)
    
    print(f"Loading GPT-2 model from {gpt2_model}...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    gpt2_model_obj = GPT2LMHeadModel.from_pretrained(gpt2_model).to(device)
    gpt2_model_obj.eval()
    
    gemma_vocab_size = len(gemma_tokenizer)
    print(f"Gemma vocabulary size: {gemma_vocab_size}")
    print(f"GPT-2 vocabulary size: {len(gpt2_tokenizer)}")
    
    # Initialize array to store log probabilities
    log_probs_np = np.zeros(gemma_vocab_size, dtype=np.float32)
    
    print("\nComputing GPT-2 probabilities for each Gemma token...")
    for token_id in tqdm(range(gemma_vocab_size)):
        # Decode Gemma token to text
        try:
            text = gemma_tokenizer.decode([token_id])
        except:
            # Some tokens might not be decodable
            log_probs_np[token_id] = float('-inf')
            continue
        
        # Compute GPT-2 probability for this text
        log_prob = compute_token_probability_gpt2(text, gpt2_model_obj, gpt2_tokenizer, device)
        log_probs_np[token_id] = log_prob
    
    print(f"\nLog probabilities computed. Shape: {log_probs_np.shape}")
    print(f"Min log prob: {log_probs_np.min():.4f}")
    print(f"Max log prob: {log_probs_np.max():.4f}")
    print(f"Mean log prob: {log_probs_np.mean():.4f}")
    
    # Show top 10 most likely tokens
    top_indices = np.argsort(log_probs_np)[-10:][::-1]
    print("\nTop 10 Gemma tokens with highest GPT-2 probability:")
    for idx in top_indices:
        token_text = gemma_tokenizer.decode([idx])
        print(f"  Gemma token {idx:6d} ({repr(token_text):20s}): log_prob = {log_probs_np[idx]:.4f}")
    
    return log_probs_np, gemma_tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute GPT-2 token probabilities for Gemma vocabulary")
    parser.add_argument('--gemma-model', type=str, default='google/gemma-2-2b', 
                        help='Gemma model name for vocabulary (default: google/gemma-2-2b)')
    parser.add_argument('--gpt2-model', type=str, default='gpt2', 
                        help='GPT-2 model name for computing probabilities (default: gpt2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--output', type=str, 
                        default='gpt2_vocab_logprobs.npy',
                        help='Output file path (default: gpt2_vocab_logprobs.npy)')
    
    args = parser.parse_args()
    
    log_probs, gemma_tokenizer = compute_vocab_probabilities(
        gemma_model=args.gemma_model,
        gpt2_model=args.gpt2_model,
        device=args.device
    )
    
    output_path = Path(args.output)
    print(f"\nSaving to {output_path}...")
    np.save(output_path, log_probs)
    
    # Also save the tokenizer info
    info_path = output_path.with_suffix('.info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Gemma model: {args.gemma_model}\n")
        f.write(f"GPT-2 model: {args.gpt2_model}\n")
        f.write(f"Gemma vocabulary size: {len(gemma_tokenizer)}\n")
        f.write(f"Shape: {log_probs.shape}\n")
        f.write(f"Min log prob: {log_probs.min():.4f}\n")
        f.write(f"Max log prob: {log_probs.max():.4f}\n")
        f.write(f"Mean log prob: {log_probs.mean():.4f}\n")
    
    print(f"Done! Saved log probabilities to {output_path}")
    print(f"Saved metadata to {info_path}")

