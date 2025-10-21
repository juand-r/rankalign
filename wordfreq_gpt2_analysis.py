#!/usr/bin/env python3
"""
Analyze correlation between word frequencies (wordfreq) and LM log-probabilities.
Two versions:
1. No prompt (just EOS token)
2. With Zipfian prompt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wordfreq import word_frequency
import seaborn as sns
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")

def get_common_words(n=5000, lang='en'):
    """Get common English words from wordfreq."""
    # Import the wordlist iterator
    from wordfreq import iter_wordlist
    
    # Get top n words from the wordlist (already sorted by frequency)
    words = list(iter_wordlist(lang, wordlist='best'))[:n]
    
    # Get their frequencies using word_frequency
    freqs = [word_frequency(word, lang, wordlist='best', minimum=0.0) for word in words]
    
    return words, freqs


def get_lm_logprob_batch(model, tokenizer, words, prompt=None, use_chat_template=False):
    """
    Get log-probabilities for a batch of words under a language model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        words: list of words to score
        prompt: optional prompt string (if None, uses BOS token)
        use_chat_template: whether to use chat template for instruction-tuned models
    
    Returns:
        list of log probabilities (floats)
    """
    all_logprobs = []
    
    for word in words:
        # Tokenize the word
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        
        if len(word_tokens) == 0:
            all_logprobs.append(np.nan)
            continue
        
        # Prepare input
        if prompt is None:
            # Use BOS token if available
            if tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[tokenizer.bos_token_id] + word_tokens])
                start_pos = 1
            else:
                input_ids = torch.tensor([word_tokens])
                start_pos = 0
        else:
            if use_chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                input_ids = torch.tensor([prompt_tokens + word_tokens])
                start_pos = len(prompt_tokens)
            else:
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
                input_ids = torch.tensor([prompt_tokens + word_tokens])
                start_pos = len(prompt_tokens)
        
        # Get logits
        with torch.no_grad():
            input_ids = input_ids.to(model.device)
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
        
        # Calculate log probabilities for the word tokens
        log_probs = []
        for i, token_id in enumerate(word_tokens):
            token_logits = logits[0, start_pos + i - 1, :]
            log_probs_dist = torch.log_softmax(token_logits, dim=-1)
            log_probs.append(log_probs_dist[token_id].item())
        
        total_log_prob = sum(log_probs)
        all_logprobs.append(total_log_prob)
    
    return all_logprobs


def get_lm_logprob(model, tokenizer, word, prompt=None, use_chat_template=False):
    """
    Get log-probability of a word under a language model.
    If word has multiple tokens, sum their log-probs.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        word: word to score
        prompt: optional prompt string (if None, uses BOS token)
        use_chat_template: whether to use chat template for instruction-tuned models
    
    Returns:
        log probability (float)
    """
    # Tokenize the word
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    
    if len(word_tokens) == 0:
        return np.nan
    
    # Prepare input
    if prompt is None:
        # Use BOS token if available, otherwise just the word tokens
        if tokenizer.bos_token_id is not None:
            input_ids = torch.tensor([[tokenizer.bos_token_id] + word_tokens])
            start_pos = 1
        else:
            input_ids = torch.tensor([word_tokens])
            start_pos = 0
    else:
        if use_chat_template:
            # Use chat template for instruction-tuned models
            # Format as a conversation where assistant should complete with the word
            messages = [
                {"role": "user", "content": prompt},
            ]
            # Apply chat template to get the prompt, then we'll append the word
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            input_ids = torch.tensor([prompt_tokens + word_tokens])
            start_pos = len(prompt_tokens)
        else:
            # Traditional approach
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([prompt_tokens + word_tokens])
            start_pos = len(prompt_tokens)
    
    # Get logits - move input to model's device
    with torch.no_grad():
        input_ids = input_ids.to(model.device)
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits
    
    # Calculate log probabilities for the word tokens
    log_probs = []
    
    for i, token_id in enumerate(word_tokens):
        # Get logits at the position before this token
        token_logits = logits[0, start_pos + i - 1, :]
        # Convert to log probabilities
        log_probs_dist = torch.log_softmax(token_logits, dim=-1)
        # Get log prob of the actual token
        log_probs.append(log_probs_dist[token_id].item())
    
    # Sum log probs for multi-token words
    total_log_prob = sum(log_probs)
    
    return total_log_prob


def analyze_correlation(words, word_freqs, gpt2_logprobs, title):
    """Calculate and print correlation statistics."""
    # Filter out any NaN values
    valid_mask = ~np.isnan(gpt2_logprobs)
    words_clean = [w for w, m in zip(words, valid_mask) if m]
    word_freqs_clean = np.array([f for f, m in zip(word_freqs, valid_mask) if m])
    logprobs_clean = np.array([lp for lp, m in zip(gpt2_logprobs, valid_mask) if m])
    
    # Log-transform word frequencies
    log_word_freqs = np.log10(word_freqs_clean)
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(log_word_freqs, logprobs_clean)
    spearman_corr, spearman_p = spearmanr(log_word_freqs, logprobs_clean)
    
    print(f"\n{title}")
    print("="*60)
    print(f"Number of words: {len(words_clean)}")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    return words_clean, log_word_freqs, logprobs_clean, pearson_corr, spearman_corr


def plot_correlation(log_word_freqs, gpt2_logprobs, title, pearson_corr, spearman_corr, filename):
    """Create scatter plot of word frequency vs GPT-2 log-prob."""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(log_word_freqs, gpt2_logprobs, alpha=0.5, s=20)
    
    # Add regression line
    z = np.polyfit(log_word_freqs, gpt2_logprobs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(log_word_freqs.min(), log_word_freqs.max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Linear fit')
    
    plt.xlabel('Log10(Word Frequency)', fontsize=12)
    plt.ylabel('GPT-2 Log-Probability', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add correlation stats to plot
    textstr = f'Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


def main():
    model_name = "google/gemma-2-2b-it"
    batch_size = 32
    print(f"Loading {model_name} model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    print("Getting common English words...")
    words, word_freqs = get_common_words(n=5000)
    print(f"Loaded {len(words)} words")
    
    # Version 1: No prompt (just BOS/EOS)
    print("\n" + "="*60)
    print("VERSION 1: No prompt (BOS token only)")
    print("="*60)
    
    lm_logprobs_v1 = []
    for i in tqdm(range(0, len(words), batch_size), desc="Processing batches (no prompt)"):
        batch = words[i:i+batch_size]
        batch_logprobs = get_lm_logprob_batch(model, tokenizer, batch, prompt=None)
        lm_logprobs_v1.extend(batch_logprobs)
    
    words_v1, log_freqs_v1, logprobs_v1, pearson_v1, spearman_v1 = analyze_correlation(
        words, word_freqs, lm_logprobs_v1, 
        "Version 1: No Prompt"
    )
    
    plot_correlation(
        log_freqs_v1, logprobs_v1,
        "Word Frequency vs Gemma-2-2b Log-Probability\n(No Prompt)",
        pearson_v1, spearman_v1,
        "wordfreq_gemma2_no_prompt.png"
    )
    
    # Version 2: With Zipfian prompt
    print("\n" + "="*60)
    print("VERSION 2: With Zipfian prompt")
    print("="*60)
    
    prompt = "English words follow a Zipfian distribution. Here is a random word:"
    
    lm_logprobs_v2 = []
    for i in tqdm(range(0, len(words), batch_size), desc="Processing batches (Zipfian prompt)"):
        batch = words[i:i+batch_size]
        batch_logprobs = get_lm_logprob_batch(model, tokenizer, batch, prompt=prompt, use_chat_template=True)
        lm_logprobs_v2.extend(batch_logprobs)
    
    words_v2, log_freqs_v2, logprobs_v2, pearson_v2, spearman_v2 = analyze_correlation(
        words, word_freqs, lm_logprobs_v2,
        "Version 2: With Zipfian Prompt"
    )
    
    plot_correlation(
        log_freqs_v2, logprobs_v2,
        "Word Frequency vs Gemma-2-2b Log-Probability\n(With Zipfian Prompt)",
        pearson_v2, spearman_v2,
        "wordfreq_gemma2_zipfian_prompt.png"
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"Version 1 (No Prompt):")
    print(f"  Pearson: {pearson_v1:.4f}, Spearman: {spearman_v1:.4f}")
    print(f"Version 2 (Zipfian Prompt):")
    print(f"  Pearson: {pearson_v2:.4f}, Spearman: {spearman_v2:.4f}")
    print(f"\nDifference in Pearson: {pearson_v2 - pearson_v1:+.4f}")
    print(f"Difference in Spearman: {spearman_v2 - spearman_v1:+.4f}")


if __name__ == "__main__":
    main()

