"""
Unified hypernym prediction generator with multiple sampling strategies.

Combines all compute_car_*.py scripts into one configurable pipeline.

Usage:
    python compute_hypernym_predictions_unified.py \\
        --noun1 cars \\
        --1token --2token --beam --handcrafted \\
        --output hypernym_predictions/cars_predictions.csv

Strategies:
    --1token:       Sample all single-token nouns
    --2token:       Sample 2-token noun phrases (top-K x top-M)
    --3token:       Sample 3-token noun phrases (top-K x top-M x top-N)
    --beam:         Beam search for variable-length sequences
    --handcrafted:  Include hand-crafted test completions from file
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import spacy
import argparse
import re
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
from nltk.corpus import words
# import enchant  # Commented out - requires C library installation

# Make langdetect deterministic
DetectorFactory.seed = 0

# Initialize English word set from NLTK (much faster as a set)
ENGLISH_WORDS = set(w.lower() for w in words.words())

# Initialize English dictionary for spell checking (commented out - requires enchant C library)
# ENGLISH_DICT = enchant.Dict("en_US")


class HypernymPredictor:
    """Unified hypernym prediction with multiple sampling strategies"""
    
    # Prompt template - modify this to change the prompt format for all strategies
    #PROMPT_TEMPLATE = "Complete the sentence: {noun1} are a kind of"
    PROMPT_TEMPLATE = "Complete the sentence: apples are a kind of fruit. Complete the sentence: {noun1} are a kind of"
    
    def __init__(self, model_name="google/gemma-2-2b", device=None):
        """Initialize model and tokenizer"""
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {self.device}")
        print(f"Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print(f"Model loaded on: {self.model.device}")
        
        # Precompute word boundary tokens
        print("Identifying word-boundary tokens...")
        self.word_start_tokens = self._get_word_boundary_tokens()
        print(f"Found {len(self.word_start_tokens)} word-starting tokens")
    
    def _get_prompt(self, noun1):
        """Get the prompt for a given noun1. Modify PROMPT_TEMPLATE to change prompt format."""
        return self.PROMPT_TEMPLATE.format(noun1=noun1)
    
    def _get_word_boundary_tokens(self):
        """Get all token IDs that start with ▁ (space/word boundary)"""
        word_start_tokens = []
        for token_id in range(len(self.tokenizer)):
            try:
                token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                if token_str.startswith('▁'):
                    word_start_tokens.append(token_id)
            except:
                continue
        return word_start_tokens
    
    def _is_valid_noun(self, word):
        """Check if word is a valid noun using spaCy"""
        word = word.strip()
        if not word or len(word) < 2:
            return False
        if not word.replace('-', '').replace("'", "").isalpha():
            return False
        
        doc = self.nlp(word)
        if len(doc) == 0:
            return False
        
        return doc[0].pos_ in ['NOUN', 'PROPN']
    
    def _is_noun_phrase(self, text):
        """Check if text is a valid noun phrase"""
        text = text.strip()
        if not text or len(text) < 2:
            return False
        
        if not any(c.isalpha() for c in text):
            return False
        
        doc = self.nlp(text)
        if len(doc) == 0:
            return False
        
        # Check if last token is noun (most common pattern)
        if doc[-1].pos_ in ['NOUN', 'PROPN']:
            return True
        
        # Check for noun chunks
        if len(list(doc.noun_chunks)) > 0:
            return True
        
        return False
    
    def _get_end_tokens(self):
        """Get token IDs for period and other end-of-phrase markers"""
        end_strings = ['.', ' .', '!', ' !', '?', ' ?', ',', ' ,', ';', ' ;']
        end_tokens = []
        for s in end_strings:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            end_tokens.extend(tokens)
        # Also add EOS token if it exists
        if self.tokenizer.eos_token_id is not None:
            end_tokens.append(self.tokenizer.eos_token_id)
        return list(set(end_tokens))  # Remove duplicates
    
    def _is_likely_complete(self, prompt_with_prediction, end_threshold=0.3):
        """
        Check if the model is likely to END (with period/punctuation) after the prediction.
        
        Returns True if next token is likely to be an end token (period, etc.)
        Returns False if next token is likely to be a continuation word
        
        Args:
            prompt_with_prediction: The prompt + predicted completion (1, 2, or 3 tokens)
            end_threshold: If P(end tokens) > this, consider it complete
        """
        # Get logits for next token after the prediction
        inputs = self.tokenizer(prompt_with_prediction, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Next token logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=0)
        
        # Check probability of end tokens (period, etc.)
        end_tokens = self._get_end_tokens()
        end_token_probs = probs[end_tokens]
        total_end_prob = end_token_probs.sum().item()
        
        # If high probability of ending, this is complete
        if total_end_prob > end_threshold:
            return True  # Likely to end (complete)
        
        return False  # Likely to continue (incomplete)
    
    def compute_single_token_predictions(self, noun1, filter_incomplete=True, end_threshold=0.3, top_k=None):
        """
        Strategy 1: All single-token nouns
        
        Args:
            noun1: The hyponym
            filter_incomplete: If True, filter out tokens where next token is unlikely to be period/end
            end_threshold: Threshold for filtering (P(end token) must be > this to keep)
            top_k: If specified, only sample top K single tokens by probability (default: None = all tokens)
        """
        print(f"\n{'='*60}")
        print("STRATEGY 1: Single-Token Nouns")
        print(f"{'='*60}")
        print(f"Filter incomplete: {filter_incomplete} (end threshold: {end_threshold})")
        if top_k:
            print(f"Sampling top-{top_k} tokens only")
        else:
            print(f"Sampling ALL tokens ({len(self.word_start_tokens)} word-starting tokens)")
        
        prompt = self._get_prompt(noun1)
        print(f"Prompt: '{prompt}'")
        
        # Get logits
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        # Get log probabilities for word-starting tokens
        word_start_logits = logits[self.word_start_tokens]
        log_probs = torch.log_softmax(word_start_logits, dim=0)
        sorted_indices = torch.argsort(log_probs, descending=True)
        
        # Optionally limit to top-K
        if top_k is not None and top_k > 0:
            sorted_indices = sorted_indices[:top_k]
        
        # Filter for valid nouns
        predictions = []
        filtered_incomplete = 0
        
        for idx in tqdm(sorted_indices, desc="Processing single tokens"):
            token_id = self.word_start_tokens[idx.item()]
            word = self.tokenizer.decode([token_id]).strip()
            log_prob = log_probs[idx].item()
            
            if self._is_valid_noun(word):
                # Check if next token after this word is likely to be period/end
                if filter_incomplete:
                    prompt_with_token = prompt + " " + word
                    if not self._is_likely_complete(prompt_with_token, end_threshold):
                        filtered_incomplete += 1
                        continue  # Skip - next token unlikely to be a period
                
                predictions.append({
                    'noun1': noun1,
                    'predicted_hypernym': word,
                    'log_prob': log_prob,
                    'num_tokens': 1,
                    'strategy': 'single_token'
                })
        
        print(f"✓ Found {len(predictions)} single-token nouns")
        if filter_incomplete:
            print(f"✓ Filtered out {filtered_incomplete} incomplete tokens (next token unlikely to be period)")
        return predictions
    
    def compute_2token_predictions(self, noun1, top_k_token1=1000, top_m_token2=500, filter_incomplete=True, end_threshold=0.3):
        """
        Strategy 2: 2-token noun phrases
        
        Args:
            filter_incomplete: If True, check if token after 2-token phrase is likely to be period/end
            end_threshold: Threshold for filtering (P(end token) must be > this to keep)
        """
        print(f"\n{'='*60}")
        print("STRATEGY 2: 2-Token Noun Phrases")
        print(f"{'='*60}")
        print(f"Configuration: Top-{top_k_token1} token1 × Top-{top_m_token2} token2")
        print(f"Filter incomplete: {filter_incomplete} (end threshold: {end_threshold})")
        
        prompt = self._get_prompt(noun1)
        
        # Step 1: Get top-K first tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits1 = outputs.logits[0, -1, :]
        
        word_start_logits = logits1[self.word_start_tokens]
        log_probs1 = torch.log_softmax(word_start_logits, dim=0)
        top_k_idx = torch.topk(log_probs1, k=min(top_k_token1, len(self.word_start_tokens))).indices
        
        top_k_token1_ids = [self.word_start_tokens[idx.item()] for idx in top_k_idx]
        top_k_log_probs1 = [log_probs1[idx].item() for idx in top_k_idx]
        
        # Step 2: For each token1, get top-M token2
        predictions = []
        filtered_incomplete = 0
        
        for token1_id, log_prob1 in tqdm(zip(top_k_token1_ids, top_k_log_probs1),
                                          total=len(top_k_token1_ids),
                                          desc="Processing 2-token combinations"):
            
            prompt_with_token1 = prompt + self.tokenizer.decode([token1_id])
            
            inputs = self.tokenizer(prompt_with_token1, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits2 = outputs.logits[0, -1, :]
            
            word_start_logits2 = logits2[self.word_start_tokens]
            log_probs2 = torch.log_softmax(word_start_logits2, dim=0)
            top_m_idx = torch.topk(log_probs2, k=min(top_m_token2, len(self.word_start_tokens))).indices
            
            for idx2 in top_m_idx:
                token2_id = self.word_start_tokens[idx2.item()]
                log_prob2 = log_probs2[idx2].item()
                
                phrase = self.tokenizer.decode([token1_id, token2_id]).strip()
                
                if self._is_noun_phrase(phrase):
                    # Check if token after this 2-token phrase is likely to be period/end
                    if filter_incomplete:
                        prompt_with_phrase = prompt + " " + phrase
                        if not self._is_likely_complete(prompt_with_phrase, end_threshold):
                            filtered_incomplete += 1
                            continue  # Skip - next token unlikely to be a period
                    
                    joint_log_prob = log_prob1 + log_prob2
                    predictions.append({
                        'noun1': noun1,
                        'predicted_hypernym': phrase,
                        'log_prob': joint_log_prob,
                        'num_tokens': 2,
                        'strategy': '2token'
                    })
        
        print(f"✓ Found {len(predictions)} 2-token noun phrases")
        if filter_incomplete:
            print(f"✓ Filtered out {filtered_incomplete} incomplete phrases (next token unlikely to be period)")
        return predictions
    
    def compute_3token_predictions(self, noun1, top_k_token1=500, top_m_token2=200, top_n_token3=100, filter_incomplete=True, end_threshold=0.3):
        """
        Strategy 3: 3-token noun phrases
        
        Args:
            filter_incomplete: If True, check if token after 3-token phrase is likely to be period/end
            end_threshold: Threshold for filtering (P(end token) must be > this to keep)
        """
        print(f"\n{'='*60}")
        print("STRATEGY 3: 3-Token Noun Phrases")
        print(f"{'='*60}")
        print(f"Configuration: Top-{top_k_token1} × Top-{top_m_token2} × Top-{top_n_token3}")
        print(f"Filter incomplete: {filter_incomplete} (end threshold: {end_threshold})")
        
        prompt = self._get_prompt(noun1)
        
        # Step 1: Get top-K first tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits1 = outputs.logits[0, -1, :]
        
        word_start_logits = logits1[self.word_start_tokens]
        log_probs1 = torch.log_softmax(word_start_logits, dim=0)
        top_k_idx = torch.topk(log_probs1, k=min(top_k_token1, len(self.word_start_tokens))).indices
        
        top_k_token1_ids = [self.word_start_tokens[idx.item()] for idx in top_k_idx]
        top_k_log_probs1 = [log_probs1[idx].item() for idx in top_k_idx]
        
        # Step 2-3: For each token1, get token2, then token3
        predictions = []
        filtered_incomplete = 0
        
        for token1_id, log_prob1 in tqdm(zip(top_k_token1_ids, top_k_log_probs1),
                                          total=len(top_k_token1_ids),
                                          desc="Processing 3-token combinations"):
            
            prompt_with_token1 = prompt + self.tokenizer.decode([token1_id])
            
            inputs = self.tokenizer(prompt_with_token1, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits2 = outputs.logits[0, -1, :]
            
            word_start_logits2 = logits2[self.word_start_tokens]
            log_probs2 = torch.log_softmax(word_start_logits2, dim=0)
            top_m_idx = torch.topk(log_probs2, k=min(top_m_token2, len(self.word_start_tokens))).indices
            
            for idx2 in top_m_idx:
                token2_id = self.word_start_tokens[idx2.item()]
                log_prob2 = log_probs2[idx2].item()
                
                prompt_with_token12 = prompt_with_token1 + self.tokenizer.decode([token2_id])
                
                inputs = self.tokenizer(prompt_with_token12, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits3 = outputs.logits[0, -1, :]
                
                word_start_logits3 = logits3[self.word_start_tokens]
                log_probs3 = torch.log_softmax(word_start_logits3, dim=0)
                top_n_idx = torch.topk(log_probs3, k=min(top_n_token3, len(self.word_start_tokens))).indices
                
                for idx3 in top_n_idx:
                    token3_id = self.word_start_tokens[idx3.item()]
                    log_prob3 = log_probs3[idx3].item()
                    
                    phrase = self.tokenizer.decode([token1_id, token2_id, token3_id]).strip()
                    
                    if self._is_noun_phrase(phrase):
                        # Check if token after this 3-token phrase is likely to be period/end
                        if filter_incomplete:
                            prompt_with_phrase = prompt + " " + phrase
                            if not self._is_likely_complete(prompt_with_phrase, end_threshold):
                                filtered_incomplete += 1
                                continue  # Skip - next token unlikely to be a period
                        
                        joint_log_prob = log_prob1 + log_prob2 + log_prob3
                        predictions.append({
                            'noun1': noun1,
                            'predicted_hypernym': phrase,
                            'log_prob': joint_log_prob,
                            'num_tokens': 3,
                            'strategy': '3token'
                        })
        
        print(f"✓ Found {len(predictions)} 3-token noun phrases")
        if filter_incomplete:
            print(f"✓ Filtered out {filtered_incomplete} incomplete phrases (next token unlikely to be period)")
        return predictions
    
    def compute_beam_search_predictions(self, noun1, num_beams=2000, num_return_sequences=2000, max_new_tokens=5):
        """Strategy 4: Beam search for variable-length sequences"""
        print(f"\n{'='*60}")
        print("STRATEGY 4: Beam Search")
        print(f"{'='*60}")
        print(f"Configuration: {num_beams} beams, {num_return_sequences} sequences, max {max_new_tokens} tokens")
        
        prompt = self._get_prompt(noun1)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=False,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Process sequences
        predictions = []
        seen_phrases = set()
        prompt_length = inputs['input_ids'].shape[1]
        
        for i, sequence in enumerate(tqdm(outputs.sequences, desc="Processing beam sequences")):
            generated_ids = sequence[prompt_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Remove everything after first punctuation
            for punct in ['.', ',', '!', '?', ';', ':', '\n', '(', ')']:
                if punct in generated_text:
                    generated_text = generated_text.split(punct)[0].strip()
            
            if not generated_text or generated_text in seen_phrases:
                continue
            
            if self._is_noun_phrase(generated_text):
                # Count tokens
                actual_tokens = self.tokenizer.encode(generated_text, add_special_tokens=False)
                num_tokens = len(actual_tokens)
                
                # Get log probability
                if hasattr(outputs, 'sequences_scores'):
                    log_prob = outputs.sequences_scores[i].item()
                else:
                    log_prob = 0.0
                    if hasattr(outputs, 'scores') and outputs.scores:
                        for step_idx, step_scores in enumerate(outputs.scores):
                            if step_idx < len(generated_ids):
                                token_id = generated_ids[step_idx]
                                log_probs = torch.log_softmax(step_scores[i], dim=0)
                                log_prob += log_probs[token_id].item()
                
                predictions.append({
                    'noun1': noun1,
                    'predicted_hypernym': generated_text,
                    'log_prob': log_prob,
                    'num_tokens': num_tokens,
                    'strategy': 'beam_search'
                })
                seen_phrases.add(generated_text)
        
        print(f"✓ Found {len(predictions)} unique beam search predictions")
        return predictions
    
    def compute_handcrafted_predictions(self, noun1, handcrafted_file):
        """Strategy 5: Hand-crafted test completions"""
        print(f"\n{'='*60}")
        print("STRATEGY 5: Hand-Crafted Test Completions")
        print(f"{'='*60}")
        
        if not Path(handcrafted_file).exists():
            print(f"⚠️  File not found: {handcrafted_file}")
            return []
        
        # Load completions
        completions = []
        current_category = None
        
        with open(handcrafted_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    current_category = line[1:].strip()
                    continue
                completions.append({
                    'completion': line,
                    'category': current_category if current_category else 'Uncategorized'
                })
        
        print(f"Loaded {len(completions)} test completions")
        
        # Compute log probabilities
        prompt = self._get_prompt(noun1)
        predictions = []
        
        for item in tqdm(completions, desc="Computing handcrafted log probs"):
            completion_text = item['completion']
            full_text = prompt + " " + completion_text
            
            # Tokenize
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.model.device)
            full_tokens = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.model.device)
            
            prompt_length = prompt_tokens.shape[1]
            completion_tokens = full_tokens[:, prompt_length:]
            
            # Compute log probability
            with torch.no_grad():
                outputs = self.model(full_tokens)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                
                total_log_prob = 0.0
                for i in range(completion_tokens.shape[1]):
                    token_id = completion_tokens[0, i].item()
                    position = prompt_length + i - 1
                    token_log_prob = log_probs[0, position, token_id].item()
                    total_log_prob += token_log_prob
            
            predictions.append({
                'noun1': noun1,
                'predicted_hypernym': completion_text,
                'log_prob': total_log_prob,
                'num_tokens': completion_tokens.shape[1],
                'strategy': 'handcrafted',
                'category': item['category']
            })
        
        print(f"✓ Computed {len(predictions)} handcrafted predictions")
        return predictions


def is_garbage(text):
    """Check if a predicted hypernym is garbage (from combine_and_clean_predictions.py)"""
    text = str(text).strip()
    
    # Empty or too short
    if len(text) < 2:
        return True
    
    # Contains any underscore
    if '_' in text:
        return True
    
    # Contains problematic characters
    if any(char in text for char in [',', '.', '=', ':', '(', '[', "'", '"', '!', '?', ';', ':', '\n', '(', ')']):
        return True
    
    # HTML tags
    if re.search(r'<[^>]+>', text):
        return True
    
    # Starts with punctuation or special chars
    if text[0] in '<>[]{}()_-=+*&^%$#@!~`':
        return True
    
    # Contains "Complete the sentence" or similar prompt repetitions
    if 'Complete the sentence' in text or 'sentence' in text.lower():
        return True
    
    # Math/XML namespaces
    if 'xmlns' in text or 'http://' in text:
        return True
    
    # All non-alphabetic (except spaces)
    if not any(c.isalpha() for c in text):
        return True
    
    # Check if text is English using BOTH NLTK words and langdetect
    # If EITHER one flags it as English, we keep it (conservative approach)
    is_english_nltk = False
    is_english_langdetect = False
    
    # Check with NLTK words corpus
    try:
        # For multi-word phrases, check each word
        text_words = text.split()
        if len(text_words) == 1:
            is_english_nltk = text.lower() in ENGLISH_WORDS
        else:
            # For phrases, at least one word should be valid English
            is_english_nltk = any(word.lower() in ENGLISH_WORDS for word in text_words if len(word) > 1)
    except:
        # If NLTK check fails, default to keeping it
        is_english_nltk = True
    
    # Commented out - pyenchant version (requires enchant C library installation)
    # try:
    #     words = text.split()
    #     if len(words) == 1:
    #         is_english_enchant = ENGLISH_DICT.check(text)
    #     else:
    #         is_english_enchant = any(ENGLISH_DICT.check(word) for word in words if len(word) > 1)
    # except:
    #     is_english_enchant = True
    
    # Check with langdetect (language detection)
    try:
        detected_lang = detect(text)
        is_english_langdetect = (detected_lang == 'en')
    except LangDetectException:
        # If langdetect fails (e.g., too short, ambiguous), default to keeping it
        is_english_langdetect = True
    
    # If NEITHER method confirms it's English, filter it out
    if not is_english_nltk and not is_english_langdetect:
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Unified hypernym prediction generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single-token and 2-token predictions for "cars"
  python compute_hypernym_predictions_unified.py --noun1 cars --1token --2token --output predictions.csv
  
  # All strategies for "dogs"
  python compute_hypernym_predictions_unified.py --noun1 dogs --1token --2token --3token --beam --handcrafted --output dogs_predictions.csv
  
  # Single-token: only top 5000 tokens (faster than processing all ~80k tokens)
  python compute_hypernym_predictions_unified.py --noun1 cars --1token --top_k_single 5000 --output cars_top5k.csv
  
  # Single-token without filtering (keeps "fuel" even if unlikely to end with period)
  python compute_hypernym_predictions_unified.py --noun1 cars --1token --no_filter_incomplete --output cars_all_single.csv
  
  # Single-token with stricter filtering (threshold 0.5 means P(period) must be > 50% to keep)
  python compute_hypernym_predictions_unified.py --noun1 cars --1token --end_threshold 0.5 --output cars_complete_only.csv
  
  # 2-token with filtering (only keeps phrases where next token is likely to be period)
  python compute_hypernym_predictions_unified.py --noun1 cars --2token --end_threshold 0.4 --output cars_2token.csv
  
  # Just beam search and handcrafted
  python compute_hypernym_predictions_unified.py --noun1 cars --beam --handcrafted --handcrafted_file test_completions.txt --output cars_beam.csv
        """
    )
    
    # Required arguments
    parser.add_argument('--noun1', type=str, required=True,
                        help='The hyponym to generate hypernym predictions for (e.g., "cars", "dogs")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    
    # Strategy flags
    parser.add_argument('--1token', action='store_true',
                        help='Generate single-token noun predictions')
    parser.add_argument('--2token', action='store_true',
                        help='Generate 2-token noun phrase predictions')
    parser.add_argument('--3token', action='store_true',
                        help='Generate 3-token noun phrase predictions')
    parser.add_argument('--beam', action='store_true',
                        help='Generate predictions using beam search')
    parser.add_argument('--handcrafted', action='store_true',
                        help='Include hand-crafted test completions from file')
    
    # Configuration parameters
    parser.add_argument('--model', type=str, default='google/gemma-2-2b',
                        help='Model name (default: google/gemma-2-2b)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--handcrafted_file', type=str, default='test_completions.txt',
                        help='Path to handcrafted completions file (default: test_completions.txt)')
    
    # Strategy-specific parameters
    parser.add_argument('--filter_incomplete', action='store_true', default=True,
                        help='Filter out predictions where next token is unlikely to be period/end (default: True)')
    parser.add_argument('--no_filter_incomplete', action='store_false', dest='filter_incomplete',
                        help='Disable filtering of incomplete predictions')
    parser.add_argument('--end_threshold', type=float, default=0.3,
                        help='Threshold for end token filtering: P(period) must be > this to keep (0-1, default: 0.3)')
    parser.add_argument('--top_k_single', type=int, default=None,
                        help='For single-token strategy: only sample top K tokens by probability (default: None = all tokens)')
    parser.add_argument('--top_k_1', type=int, default=1000,
                        help='Top K tokens for 2-token strategy first position (default: 1000)')
    parser.add_argument('--top_m_2', type=int, default=500,
                        help='Top M tokens for 2-token strategy second position (default: 500)')
    parser.add_argument('--top_k_3_1', type=int, default=500,
                        help='Top K tokens for 3-token strategy first position (default: 500)')
    parser.add_argument('--top_m_3_2', type=int, default=200,
                        help='Top M tokens for 3-token strategy second position (default: 200)')
    parser.add_argument('--top_n_3_3', type=int, default=100,
                        help='Top N tokens for 3-token strategy third position (default: 100)')
    parser.add_argument('--num_beams', type=int, default=2000,
                        help='Number of beams for beam search (default: 2000)')
    parser.add_argument('--num_return_sequences', type=int, default=2000,
                        help='Number of sequences to return from beam search (default: 2000)')
    parser.add_argument('--max_new_tokens', type=int, default=5,
                        help='Maximum new tokens for beam search (default: 5)')
    
    # Output options
    parser.add_argument('--keep_garbage', action='store_true',
                        help='Keep garbage predictions in output (default: filter them out)')
    parser.add_argument('--save_garbage', type=str, default=None,
                        help='Save filtered garbage to separate CSV file')
    
    args = parser.parse_args()
    
    # Check that at least one strategy is selected
    if not any([args.__dict__['1token'], args.__dict__['2token'], args.__dict__['3token'], 
                args.beam, args.handcrafted]):
        parser.error("At least one strategy must be selected (--1token, --2token, --3token, --beam, --handcrafted)")
    
    print("="*60)
    print("UNIFIED HYPERNYM PREDICTION GENERATOR")
    print("="*60)
    print(f"Noun1: {args.noun1}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"\nActive strategies:")
    if args.__dict__['1token']:
        filter_status = "ON" if args.filter_incomplete else "OFF"
        top_k_str = f"top-{args.top_k_single}" if args.top_k_single else "all"
        print(f"  ✓ Single-token nouns ({top_k_str} tokens, filter: {filter_status}, end threshold: {args.end_threshold})")
    if args.__dict__['2token']:
        filter_status = "ON" if args.filter_incomplete else "OFF"
        print(f"  ✓ 2-token noun phrases (top-{args.top_k_1} × top-{args.top_m_2}, filter: {filter_status})")
    if args.__dict__['3token']:
        filter_status = "ON" if args.filter_incomplete else "OFF"
        print(f"  ✓ 3-token noun phrases (top-{args.top_k_3_1} × top-{args.top_m_3_2} × top-{args.top_n_3_3}, filter: {filter_status})")
    if args.beam:
        print(f"  ✓ Beam search ({args.num_beams} beams, {args.num_return_sequences} sequences)")
    if args.handcrafted:
        print(f"  ✓ Hand-crafted completions ({args.handcrafted_file})")
    print("="*60)
    
    # Initialize predictor
    predictor = HypernymPredictor(model_name=args.model, device=args.device)
    
    # Collect all predictions
    all_predictions = []
    
    if args.__dict__['1token']:
        predictions = predictor.compute_single_token_predictions(
            args.noun1,
            filter_incomplete=args.filter_incomplete,
            end_threshold=args.end_threshold,
            top_k=args.top_k_single
        )
        all_predictions.extend(predictions)
    
    if args.__dict__['2token']:
        predictions = predictor.compute_2token_predictions(
            args.noun1, 
            top_k_token1=args.top_k_1,
            top_m_token2=args.top_m_2,
            filter_incomplete=args.filter_incomplete,
            end_threshold=args.end_threshold
        )
        all_predictions.extend(predictions)
    
    if args.__dict__['3token']:
        predictions = predictor.compute_3token_predictions(
            args.noun1,
            top_k_token1=args.top_k_3_1,
            top_m_token2=args.top_m_3_2,
            top_n_token3=args.top_n_3_3,
            filter_incomplete=args.filter_incomplete,
            end_threshold=args.end_threshold
        )
        all_predictions.extend(predictions)
    
    if args.beam:
        predictions = predictor.compute_beam_search_predictions(
            args.noun1,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens
        )
        all_predictions.extend(predictions)
    
    if args.handcrafted:
        predictions = predictor.compute_handcrafted_predictions(
            args.noun1,
            args.handcrafted_file
        )
        all_predictions.extend(predictions)
    
    print(f"\n{'='*60}")
    print("COMBINING AND CLEANING")
    print(f"{'='*60}")
    print(f"Total predictions collected: {len(all_predictions)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_predictions)
    
    # Filter garbage
    if not args.keep_garbage:
        print("\nFiltering garbage predictions...")
        before_count = len(df)
        garbage_mask = df['predicted_hypernym'].apply(is_garbage)
        
        if args.save_garbage and garbage_mask.sum() > 0:
            garbage_df = df[garbage_mask].copy()
            garbage_df = garbage_df.sort_values('log_prob', ascending=False).reset_index(drop=True)
            garbage_df.to_csv(args.save_garbage, index=False)
            print(f"  ✓ Saved {len(garbage_df)} garbage predictions to {args.save_garbage}")
        
        df = df[~garbage_mask]
        after_count = len(df)
        print(f"  ✓ Removed {before_count - after_count} garbage predictions")
        print(f"  ✓ Remaining: {after_count} predictions")
    
    # Remove duplicates (keep highest log_prob)
    print("\nRemoving duplicates (keeping highest log_prob)...")
    before_count = len(df)
    df = df.sort_values('log_prob', ascending=False)
    df = df.drop_duplicates(subset=['noun1', 'predicted_hypernym'], keep='first')
    after_count = len(df)
    print(f"  ✓ Removed {before_count - after_count} duplicates")
    print(f"  ✓ Unique predictions: {after_count}")
    
    # Sort by log_prob (descending)
    df = df.sort_values('log_prob', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Saved {len(df)} predictions to {output_path}")
    print(f"\nSummary:")
    print(f"  Total unique predictions: {len(df)}")
    print(f"  Highest log_prob: {df['log_prob'].max():.4f}")
    print(f"  Lowest log_prob: {df['log_prob'].min():.4f}")
    print(f"  Mean log_prob: {df['log_prob'].mean():.4f}")
    
    if 'strategy' in df.columns:
        print(f"\nBy strategy:")
        for strategy in df['strategy'].unique():
            count = (df['strategy'] == strategy).sum()
            print(f"  {strategy:20s}: {count:6d} predictions")
    
    # Print top 30
    print(f"\nTop 30 predictions for '{args.noun1}':")
    print(f"{'Rank':<6} {'Hypernym':<40} {'Log Prob':<12} {'Strategy':<15}")
    print("-" * 75)
    for i, row in df.head(30).iterrows():
        strategy = row.get('strategy', 'unknown')
        print(f"{i+1:<6} {row['predicted_hypernym']:<40} {row['log_prob']:<12.4f} {strategy:<15}")
    
    print(f"\n{'='*60}")
    print("✅ DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

