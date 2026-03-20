#!/usr/bin/env python3
"""
End-to-end pipeline for generating balanced hypernym datasets.

Takes a noun (e.g., "cars", "fruit") and:
1. Samples completions using multiple strategies (1-token, 2-token, beam search, GPT-4 suggestions)
2. Computes generator log-probabilities
3. Gets GPT-4 ground truth labels
4. Adds typicality scores (GPT-2, word frequency)
5. Balances positives/negatives with good log-prob coverage
6. Creates train/test splits

Usage:
    python generate_hypernym_dataset.py --noun1 cars --total 4000 --output_dir ../data
    python generate_hypernym_dataset.py --noun1 fruit --total 4000 --output_dir ../data
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
import random
import time
from openai import OpenAI

# English language detection
from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet as wn

# Make langdetect deterministic
DetectorFactory.seed = 0

# Initialize English word set from NLTK (much faster as a set)
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel


class HypernymDatasetGenerator:
    """Generate balanced hypernym datasets with multiple sampling strategies."""
    
    # One-shot prompt (matches compute_hypernym_predictions_unified.py)
    # The example helps guide the model toward valid noun completions
    PROMPT_TEMPLATE = "Complete the sentence: apples are a kind of fruit. Complete the sentence: {noun1} are a kind of"
    
    def __init__(self, noun1, model_name="google/gemma-2-2b", device=None):
        self.noun1 = noun1
        self.model_name = model_name
        # Sanitize model name for use in filenames (replace / with -)
        self.model_name_safe = model_name.replace("/", "-")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"=" * 60)
        print(f"HYPERNYM DATASET GENERATOR")
        print(f"=" * 60)
        print(f"Noun: {noun1}")
        print(f"Device: {self.device}")
        
        # Load spaCy for noun validation
        print("Loading spaCy...")
        import spacy
        # Use large model for better POS tagging (small model mis-tags "mammal" as ADJ)
        self.nlp = spacy.load("en_core_web_lg")
        
        # Load main model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        # Precompute word boundary tokens (for Gemma tokenizer)
        self.word_start_tokens = self._get_word_boundary_tokens()
        print(f"Found {len(self.word_start_tokens)} word-starting tokens")
        
        # Will load GPT-2 lazily for typicality
        self.gpt2_model = None
        self.gpt2_tokenizer = None
    
    def _get_word_boundary_tokens(self):
        """Get token IDs that start with ▁ (word boundary)"""
        word_start_tokens = []
        for token_id in range(len(self.tokenizer)):
            try:
                token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                if token_str.startswith('▁'):
                    word_start_tokens.append(token_id)
            except:
                continue
        return word_start_tokens
    
    # Words that indicate instruction artifacts or meta-language (not real hypernyms)
    FILTER_WORDS = {
        'complete', 'answer', 'fill', 'read', 'continue', 'correct', 
        'example', 'explanation', 'sentence', 'question', 'ingredient',
        'download', 'click', 'select', 'choose'
    }
    
    # Standalone function words that aren't valid hypernyms
    STOPWORDS = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'my', 'me', 'i', 'we', 'he', 'she', 'it', 'they', 'you',
        'him', 'her', 'us', 'them',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'what', 'which', 'who', 'whom', 'whose',
        'and', 'or', 'but', 'so', 'if', 'then',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
        'as', 'an', 'the', 'its',
        # Common garbage single words
        'yes', 'no', 'why', 'how', 'wow', 'get', 'got', 'sth', 'etc',
        'ted', 'gin', 'hay', 'ham', 'pot', 'pen', 'bun', 'ark', 'vet',
        'job', 'way', 'den', 'red', 'top', 'shy', 'foe', 'kin', 'aid',
        'cd', 'es', 'ca'  # 2-letter garbage
    }
    
    def _is_valid_noun(self, word):
        """Check if word is a valid noun using spaCy"""
        word = word.strip()
        if not word or len(word) < 3:  # Minimum 3 chars to filter out CD, ES, CA etc.
            return False
        
        # Filter out entries ending with " too" (e.g., "animal too", "thing too")
        if word.lower().endswith(' too'):
            return False
        
        # Filter out entries with multiple dashes (e.g., "------------ fruit")
        if '--' in word:
            return False
        
        # Filter out entries starting with special characters
        if word[0] in '<>[]{}()_-=+*&^%$#@!~`':
            return False
        
        # Filter out entries starting with 1-2 chars + space (e.g., "A food", "An food", "CD player")
        parts = word.split()
        if len(parts) > 1 and len(parts[0]) <= 2:
            return False
        
        # Filter out entries where second word is 1-2 letters (e.g., "food a", "animal of")
        if len(parts) > 1 and len(parts[1]) <= 2:
            return False
        
        # Filter out non-ASCII characters (foreign language completions)
        if not all(ord(c) < 128 for c in word):
            return False
        
        # Filter out entries containing underscore (common garbage pattern)
        if '_' in word:
            return False
        
        # Filter out entries containing problematic special characters
        if any(char in word for char in ['=', ':', '(', '[', ']', ')', '<', '>', '{', '}']):
            return False
        
        # Filter out URLs and namespaces
        if 'http://' in word or 'https://' in word or 'xmlns' in word:
            return False
        
        # Filter out standalone stopwords/function words
        if word.lower() in self.STOPWORDS:
            return False
        
        # Filter out entries starting with prepositions/articles/conjunctions
        word_lower = word.lower()
        starts_with_bad = ['the ', 'of ', 'in ', 'on ', 'at ', 'to ', 'for ', 'with ', 
                           'by ', 'from ', 'and ', 'or ', 'but ', 'as ', 'is ', 'are ',
                           'what ', 'which ', 'who ', 'how ', 'why ', 'when ', 'where ']
        if any(word_lower.startswith(prefix) for prefix in starts_with_bad):
            return False
        
        # Filter out entries containing instruction/meta-language words
        if any(fw in word_lower for fw in self.FILTER_WORDS):
            return False
        
        # Must contain only ASCII letters, spaces, hyphens, apostrophes
        if not word.replace('-', '').replace("'", "").replace(' ', '').isalpha():
            return False
        
        # Check if text is English using BOTH NLTK words and langdetect
        # If EITHER one flags it as English, we keep it (conservative approach)
        is_english_nltk = False
        is_english_langdetect = False
        
        # Check with NLTK words corpus
        try:
            word_parts = word.split()
            if len(word_parts) == 1:
                is_english_nltk = word.lower() in ENGLISH_WORDS
            else:
                # For phrases, at least one word should be valid English
                is_english_nltk = any(w.lower() in ENGLISH_WORDS for w in word_parts if len(w) > 1)
        except:
            # If NLTK check fails, default to keeping it
            is_english_nltk = True
        
        # Check with langdetect (language detection)
        try:
            detected_lang = detect(word)
            is_english_langdetect = (detected_lang == 'en')
        except LangDetectException:
            # If langdetect fails (e.g., too short, ambiguous), default to keeping it
            is_english_langdetect = True
        
        # If NEITHER method confirms it's English, filter it out
        if not is_english_nltk and not is_english_langdetect:
            return False
        
        # Check if word can be a noun using EITHER spaCy OR WordNet
        # (if either classifies it as a noun, treat it as a noun)
        
        # Method 1: spaCy POS tagging
        is_noun_spacy = False
        doc = self.nlp(word)
        if len(doc) > 0:
            # Check if last token is noun
            if doc[-1].pos_ in ['NOUN', 'PROPN']:
                is_noun_spacy = True
            # Check for noun chunks
            elif len(list(doc.noun_chunks)) > 0:
                is_noun_spacy = True
        
        # Method 2: WordNet - check if word has any noun synsets
        is_noun_wordnet = False
        try:
            # For multi-word phrases, check the last word (the head noun)
            word_to_check = word.split()[-1] if ' ' in word else word
            noun_synsets = wn.synsets(word_to_check, pos=wn.NOUN)
            is_noun_wordnet = len(noun_synsets) > 0
        except:
            pass
        
        # Accept if EITHER method classifies it as a noun
        return is_noun_spacy or is_noun_wordnet
    
    def _get_prompt(self):
        return self.PROMPT_TEMPLATE.format(noun1=self.noun1)
    
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
    
    def _compute_log_prob(self, completion):
        """Compute log probability of completion given prompt"""
        prompt = self._get_prompt()
        full_text = prompt + " " + completion
        
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
        
        # Sum log probs for completion tokens
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total_log_prob = 0.0
        num_tokens = 0
        
        for i in range(prompt_len, input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            token_log_prob = log_probs[i - 1, token_id].item()
            total_log_prob += token_log_prob
            num_tokens += 1
        
        return total_log_prob, num_tokens
    
    # ===== SAMPLING STRATEGIES =====
    
    def sample_1token(self, top_k=3000, filter_incomplete=True, end_threshold=0.3, collect_garbage=False):
        """Sample single-token completions
        
        Args:
            top_k: Number of top tokens to sample
            filter_incomplete: If True, filter out tokens where model would continue (e.g., "motor" -> "motor vehicle")
            end_threshold: Threshold for filtering - P(end token) must be > this to keep
            collect_garbage: If True, return (results, garbage) tuple
        
        Returns:
            If collect_garbage: (results, garbage) where garbage contains filtered items with reason
            Otherwise: results list
        """
        print(f"\n[Strategy: 1-token] Sampling top {top_k} single-token completions...")
        if filter_incomplete:
            print(f"  Filtering incomplete phrases (end_threshold={end_threshold})")
        
        prompt = self._get_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get top-k among word-start tokens only
        word_start_mask = torch.zeros_like(log_probs, dtype=torch.bool)
        word_start_mask[self.word_start_tokens] = True
        
        masked_log_probs = log_probs.clone()
        masked_log_probs[~word_start_mask] = float('-inf')
        
        top_values, top_indices = torch.topk(masked_log_probs, min(top_k, len(self.word_start_tokens)))
        
        results = []
        garbage = []
        filtered_incomplete = 0
        
        for log_prob, token_id in tqdm(zip(top_values.tolist(), top_indices.tolist()),
                                        total=len(top_indices), desc="  1-token"):
            token = self.tokenizer.decode([token_id]).strip()
            
            if not self._is_valid_noun(token):
                if collect_garbage:
                    garbage.append({
                        'noun1': self.noun1,
                        'predicted_hypernym': token,
                        'log_prob': log_prob,
                        'num_tokens': 1,
                        'strategy': 'single_token',
                        'filter_reason': 'not_valid_noun'
                    })
                continue
            
            # Check if next token after this word is likely to be period/end
            if filter_incomplete:
                prompt_with_token = prompt + " " + token
                if not self._is_likely_complete(prompt_with_token, end_threshold):
                    filtered_incomplete += 1
                    if collect_garbage:
                        garbage.append({
                            'noun1': self.noun1,
                            'predicted_hypernym': token,
                            'log_prob': log_prob,
                            'num_tokens': 1,
                            'strategy': 'single_token',
                            'filter_reason': 'incomplete_phrase'
                        })
                    continue  # Skip - next token unlikely to be a period
            
            results.append({
                'noun1': self.noun1,
                'predicted_hypernym': token,
                'log_prob': log_prob,
                'num_tokens': 1,
                'strategy': 'single_token'
            })
        
        print(f"  Found {len(results)} valid single-token nouns")
        if filter_incomplete:
            print(f"  Filtered out {filtered_incomplete} incomplete tokens")
        if collect_garbage:
            print(f"  Collected {len(garbage)} garbage items")
            return results, garbage
        return results
    
    def sample_2token(self, top_k1=200, top_k2=50, filter_incomplete=True, end_threshold=0.3, collect_garbage=False):
        """Sample 2-token completions
        
        Args:
            top_k1: Number of top first tokens to sample
            top_k2: Number of top second tokens per first token
            filter_incomplete: If True, filter out phrases where model would continue
            end_threshold: Threshold for filtering - P(end token) must be > this to keep
            collect_garbage: If True, return (results, garbage) tuple
        """
        print(f"\n[Strategy: 2-token] Sampling {top_k1}x{top_k2} two-token completions...")
        if filter_incomplete:
            print(f"  Filtering incomplete phrases (end_threshold={end_threshold})")
        
        prompt = self._get_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
        
        log_probs1 = torch.log_softmax(logits, dim=-1)
        
        # Get top-k1 first tokens
        word_start_mask = torch.zeros_like(log_probs1, dtype=torch.bool)
        word_start_mask[self.word_start_tokens] = True
        masked_log_probs1 = log_probs1.clone()
        masked_log_probs1[~word_start_mask] = float('-inf')
        
        top_values1, top_indices1 = torch.topk(masked_log_probs1, min(top_k1, len(self.word_start_tokens)))
        
        results = []
        garbage = []
        filtered_incomplete = 0
        
        for log_prob1, token_id1 in tqdm(zip(top_values1.tolist(), top_indices1.tolist()), 
                                          total=len(top_indices1), desc="  2-token"):
            # Extend sequence
            extended_ids = torch.cat([input_ids, torch.tensor([[token_id1]], device=self.model.device)], dim=1)
            
            with torch.no_grad():
                outputs2 = self.model(extended_ids)
                logits2 = outputs2.logits[0, -1, :]
            
            log_probs2 = torch.log_softmax(logits2, dim=-1)
            masked_log_probs2 = log_probs2.clone()
            masked_log_probs2[~word_start_mask] = float('-inf')
            
            top_values2, top_indices2 = torch.topk(masked_log_probs2, min(top_k2, len(self.word_start_tokens)))
            
            for log_prob2, token_id2 in zip(top_values2.tolist(), top_indices2.tolist()):
                completion = self.tokenizer.decode([token_id1, token_id2]).strip()
                total_log_prob = log_prob1 + log_prob2
                
                if not self._is_valid_noun(completion):
                    if collect_garbage:
                        garbage.append({
                            'noun1': self.noun1,
                            'predicted_hypernym': completion,
                            'log_prob': total_log_prob,
                            'num_tokens': 2,
                            'strategy': '2token',
                            'filter_reason': 'not_valid_noun'
                        })
                    continue
                
                # Check if token after this 2-token phrase is likely to be period/end
                if filter_incomplete:
                    prompt_with_phrase = prompt + " " + completion
                    if not self._is_likely_complete(prompt_with_phrase, end_threshold):
                        filtered_incomplete += 1
                        if collect_garbage:
                            garbage.append({
                                'noun1': self.noun1,
                                'predicted_hypernym': completion,
                                'log_prob': total_log_prob,
                                'num_tokens': 2,
                                'strategy': '2token',
                                'filter_reason': 'incomplete_phrase'
                            })
                        continue  # Skip - next token unlikely to be a period
                
                results.append({
                    'noun1': self.noun1,
                    'predicted_hypernym': completion,
                    'log_prob': total_log_prob,
                    'num_tokens': 2,
                    'strategy': '2token'
                })
        
        print(f"  Found {len(results)} valid 2-token noun phrases")
        if filter_incomplete:
            print(f"  Filtered out {filtered_incomplete} incomplete phrases")
        if collect_garbage:
            print(f"  Collected {len(garbage)} garbage items")
            return results, garbage
        return results
    
    # NOTE: 3-token sampling not implemented - not needed for current use case
    # NOTE: Handcrafted/custom completions strategy not implemented for now
    
    def sample_beam_search(self, num_beams=500, max_new_tokens=5):
        """Sample using beam search"""
        print(f"\n[Strategy: Beam] Beam search with {num_beams} beams, max {max_new_tokens} tokens...")
        
        prompt = self._get_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=False,  # Don't stop at EOS - get more diverse completions
                do_sample=False,  # Deterministic beam search
                use_cache=False
            )
        
        results = []
        prompt_len = inputs.input_ids.shape[1]
        
        for i, seq in enumerate(outputs.sequences):
            completion_ids = seq[prompt_len:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            
            # Clean up: take first noun phrase (before punctuation)
            for punct in ['.', ',', '!', '?', ';', ':', '\n', '(', ')']:
                if punct in completion:
                    completion = completion.split(punct)[0].strip()
            
            if self._is_valid_noun(completion) and len(completion) > 1:
                # Use beam search sequence scores if available (more accurate)
                if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                    log_prob = outputs.sequences_scores[i].item()
                    num_tokens = len(self.tokenizer.encode(completion, add_special_tokens=False))
                else:
                    # Fall back to computing log prob
                    log_prob, num_tokens = self._compute_log_prob(completion)
                
                results.append({
                    'noun1': self.noun1,
                    'predicted_hypernym': completion,
                    'log_prob': log_prob,
                    'num_tokens': num_tokens,
                    'strategy': 'beam_search'
                })
        
        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            if r['predicted_hypernym'] not in seen:
                seen.add(r['predicted_hypernym'])
                unique_results.append(r)
        
        print(f"  Found {len(unique_results)} unique valid completions from beam search")
        return unique_results
    
    def sample_gpt4_positive(self, num_suggestions=200, num_calls=1):
        """Get positive hypernym suggestions from GPT-4 (things that X IS a kind of)
        
        Args:
            num_suggestions: Target suggestions per call
            num_calls: Number of GPT-4 calls to make (results are aggregated)
        """
        print(f"\n[Strategy: GPT-4 Positive] Getting positive hypernym suggestions ({num_calls} calls)...")
        
        client = OpenAI()
        
        prompt = f"""Give me a comprehensive list of things that {self.noun1} ARE a kind of.
Include:
- Typical/obvious categories (e.g., if {self.noun1}="cars", include "vehicle", "transportation")
- Less typical but valid categories (e.g., "machine", "possession", "product")
- Abstract categories (e.g., "object", "thing", "item")
- Technically correct but unusual categories (e.g., "investment", "asset")
- Categories that are debatable or edge cases

Each should be a singular noun or noun phrase.
Aim for at least {num_suggestions} diverse suggestions covering the full spectrum from typical to atypical."""

        all_suggestions = set()
        
        for call_idx in range(num_calls):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=4000,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "suggestions",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["items"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                data = json.loads(response.choices[0].message.content)
                suggestions = data["items"]
                new_count = len(set(suggestions) - all_suggestions)
                all_suggestions.update(suggestions)
                
                if num_calls > 1:
                    print(f"  Call {call_idx + 1}: got {len(suggestions)}, {new_count} new (total unique: {len(all_suggestions)})")
                    
            except Exception as e:
                print(f"  Error on call {call_idx + 1}: {e}")
                continue
        
        print(f"  GPT-4 suggested {len(all_suggestions)} unique positive categories")
        
        # Compute log probs for each
        results = []
        for completion in tqdm(list(all_suggestions), desc="  Computing log probs"):
            if self._is_valid_noun(completion):
                log_prob, num_tokens = self._compute_log_prob(completion)
                results.append({
                    'noun1': self.noun1,
                    'predicted_hypernym': completion,
                    'log_prob': log_prob,
                    'num_tokens': num_tokens,
                    'strategy': 'gpt4_positive'
                })
        
        print(f"  Found {len(results)} valid positive suggestions")
        return results
    
    def sample_gpt4_negative(self, num_suggestions=200, num_calls=1):
        """Get negative hypernym suggestions from GPT-4 (things that X is NOT a kind of)
        
        Args:
            num_suggestions: Target suggestions per call
            num_calls: Number of GPT-4 calls to make (results are aggregated)
        """
        print(f"\n[Strategy: GPT-4 Negative] Getting negative hypernym suggestions ({num_calls} calls)...")
        
        client = OpenAI()
        
        prompt = f"""Give me a comprehensive list of things that {self.noun1} are NOT a kind of.
Include:
- Obvious non-categories (e.g., if {self.noun1}="cars", include "animal", "food", "plant")
- Less obvious but still incorrect categories (e.g., "furniture", "clothing", "beverage")
- Categories that might seem related but are wrong (e.g., "road", "gasoline", "driver")
- Abstract categories which are not valid (e.g., "emotion", "feeling", "thought")
- Edge cases.
- Categories that are clearly wrong, but phrased in an unusual way.

Each should be a singular noun or noun phrase.
Aim for at least {num_suggestions} diverse suggestions covering both obvious and non-obvious incorrect categories."""

        all_suggestions = set()
        
        for call_idx in range(num_calls):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=4000,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "suggestions",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["items"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                data = json.loads(response.choices[0].message.content)
                suggestions = data["items"]
                new_count = len(set(suggestions) - all_suggestions)
                all_suggestions.update(suggestions)
                
                if num_calls > 1:
                    print(f"  Call {call_idx + 1}: got {len(suggestions)}, {new_count} new (total unique: {len(all_suggestions)})")
                    
            except Exception as e:
                print(f"  Error on call {call_idx + 1}: {e}")
                continue
        
        print(f"  GPT-4 suggested {len(all_suggestions)} unique negative categories")
        
        # Compute log probs for each
        results = []
        for completion in tqdm(list(all_suggestions), desc="  Computing log probs"):
            if self._is_valid_noun(completion):
                log_prob, num_tokens = self._compute_log_prob(completion)
                results.append({
                    'noun1': self.noun1,
                    'predicted_hypernym': completion,
                    'log_prob': log_prob,
                    'num_tokens': num_tokens,
                    'strategy': 'gpt4_negative'
                })
        
        print(f"  Found {len(results)} valid negative suggestions")
        return results
    
    def sample_existing_hypernyms(self):
        """
        Sample from the 44 hypernyms in the existing hypernym dataset.
        
        These are known good hypernym categories like 'food', 'animal', 'tool', etc.
        Useful for ensuring coverage of standard categories.
        """
        print(f"\n[Strategy: Existing Hypernyms] Loading hypernyms from dataset...")
        
        # Import the data loader
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from utils import load_noun_pair_data
        
        # Load all data and extract unique hypernyms (noun2)
        L = load_noun_pair_data()
        existing_hypernyms = set(item.noun2 for item in L)
        
        print(f"  Found {len(existing_hypernyms)} unique hypernyms in dataset")
        
        # Compute log probs for each
        results = []
        for completion in tqdm(sorted(existing_hypernyms), desc="  Computing log probs"):
            if self._is_valid_noun(completion):
                log_prob, num_tokens = self._compute_log_prob(completion)
                results.append({
                    'noun1': self.noun1,
                    'predicted_hypernym': completion,
                    'log_prob': log_prob,
                    'num_tokens': num_tokens,
                    'strategy': 'existing_hypernyms'
                })
        
        print(f"  Found {len(results)} valid existing hypernyms")
        return results
    
    # ===== POST-PROCESSING =====
    
    def _get_gpt4_label(self, noun1, predicted_hypernym, client, max_retries=3):
        """
        Query GPT-4.1-mini for ground truth answer with retry logic.
        
        Args:
            noun1: The hyponym (e.g., "cars")
            predicted_hypernym: The predicted hypernym (e.g., "vehicles")
            client: OpenAI client
            max_retries: Maximum number of retries on failure
        
        Returns:
            String: "Yes", "No", or "Unknown" if failed
        """
        prompt = f"Is it true that {noun1} are a kind of {predicted_hypernym}? Answer only Yes or No. Answer:"
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                answer = response.choices[0].message.content.strip()
                
                # Normalize answer to "Yes" or "No"
                answer_lower = answer.lower()
                if "yes" in answer_lower:
                    return "Yes"
                elif "no" in answer_lower:
                    return "No"
                else:
                    # If not clear Yes/No, return the raw answer
                    return answer
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n  Warning: Error on attempt {attempt + 1}: {e}")
                    print(f"  Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"\n  Failed after {max_retries} attempts: {e}")
                    return "Unknown"
        
        return "Unknown"
    
    def add_gpt4_labels(self, df, save_every=50, checkpoint_file=None):
        """Add GPT-4 ground truth labels with incremental saving"""
        print(f"\n[Labels] Getting GPT-4 ground truth labels...")
        # TODO: Add rate limiting for GPT-4 API calls
        
        client = OpenAI()
        
        # Initialize column if not exists
        if 'gpt4_ground_truth' not in df.columns:
            df['gpt4_ground_truth'] = None
        
        # Find rows that need labeling
        needs_label = df['gpt4_ground_truth'].isna() | (df['gpt4_ground_truth'] == '')
        indices_to_label = df[needs_label].index.tolist()
        
        if not indices_to_label:
            print("  All samples already labeled!")
            return df
        
        print(f"  {len(indices_to_label)} samples need labeling...")
        
        for i, idx in enumerate(tqdm(indices_to_label, desc="  GPT-4 labels")):
            row = df.loc[idx]
            label = self._get_gpt4_label(self.noun1, row["predicted_hypernym"], client)
            df.loc[idx, 'gpt4_ground_truth'] = label
            
            # Save checkpoint every N samples
            if checkpoint_file and (i + 1) % save_every == 0:
                df.to_csv(checkpoint_file, index=False)
                # Don't print every time - just save silently
        
        # Final save
        if checkpoint_file:
            df.to_csv(checkpoint_file, index=False)
        
        return df
    
    def _compute_gpt2_token_probability(self, text):
        """
        Compute the unconditional probability of text under GPT-2.
        For multi-token text, returns the joint probability (product of token probs).
        
        This properly handles single-token words by using BOS context.
        Returns log probability.
        """
        with torch.no_grad():
            # Tokenize without special tokens to get just the content tokens
            input_ids = self.gpt2_tokenizer.encode(text, add_special_tokens=False)
            
            if len(input_ids) == 0:
                return float('-inf')
            
            # For a single token, compute P(token | BOS)
            if len(input_ids) == 1:
                # Use empty context (just BOS if model has it)
                context_ids = self.gpt2_tokenizer.encode("", add_special_tokens=True)
                full_ids = context_ids + input_ids
                
                input_tensor = torch.tensor([full_ids]).to(self.device)
                outputs = self.gpt2_model(input_tensor)
                logits = outputs.logits
                
                # Get probability of the target token given BOS
                target_logits = logits[0, len(context_ids) - 1, :]
                probs = torch.softmax(target_logits, dim=-1)
                token_prob = probs[input_ids[0]].item()
                
                return np.log(token_prob + 1e-12)
            
            # For multi-token text, compute product of conditional probabilities
            # P(t1, t2, t3) = P(t1) * P(t2|t1) * P(t3|t1,t2)
            log_prob_sum = 0.0
            
            for i in range(len(input_ids)):
                # Context is all tokens before position i
                if i == 0:
                    context_ids = self.gpt2_tokenizer.encode("", add_special_tokens=True)
                else:
                    context_ids = self.gpt2_tokenizer.encode("", add_special_tokens=True)[:-1] + input_ids[:i]
                
                full_ids = context_ids + [input_ids[i]]
                input_tensor = torch.tensor([full_ids]).to(self.device)
                outputs = self.gpt2_model(input_tensor)
                logits = outputs.logits
                
                # Get probability of token i given context
                target_logits = logits[0, len(context_ids) - 1, :]
                probs = torch.softmax(target_logits, dim=-1)
                token_prob = probs[input_ids[i]].item()
                
                log_prob_sum += np.log(token_prob + 1e-12)
            
            return log_prob_sum
    
    def _compute_gpt2_conditional_probability(self, target, context):
        """
        Compute P(target | context) under GPT-2.
        
        Args:
            target: The text to compute probability for (e.g., "vehicles")
            context: The conditioning context (e.g., "cars are a kind of")
            
        Returns log probability.
        """
        with torch.no_grad():
            # Tokenize context and target separately
            context_ids = self.gpt2_tokenizer.encode(context, add_special_tokens=True)
            target_ids = self.gpt2_tokenizer.encode(target, add_special_tokens=False)
            
            if len(target_ids) == 0:
                return float('-inf')
            
            # Compute P(target | context) = product of P(target_token_i | context + target_tokens[:i])
            log_prob_sum = 0.0
            
            for i in range(len(target_ids)):
                # Full input is context + target tokens up to and including position i
                full_ids = context_ids + target_ids[:i+1]
                input_tensor = torch.tensor([full_ids]).to(self.device)
                
                outputs = self.gpt2_model(input_tensor)
                logits = outputs.logits
                
                # Get probability of target_ids[i] given everything before it
                target_position = len(context_ids) + i - 1
                target_logits = logits[0, target_position, :]
                probs = torch.softmax(target_logits, dim=-1)
                token_prob = probs[target_ids[i]].item()
                
                log_prob_sum += np.log(token_prob + 1e-12)
            
            return log_prob_sum
    
    def add_typicality_scores(self, df):
        """Add GPT-2 typicality and word frequency scores
        
        Adds the following columns:
        - log_prob_noun2: P(noun2) unconditional GPT-2 probability of hypernym
        - log_prob_noun1: P(noun1) unconditional GPT-2 probability of hyponym
        - log_prob_noun2_given_context: P(noun2 | "{noun1} are a kind of") conditional
        - log_wordfreq_noun2: corpus frequency of hypernym
        - log_wordfreq_noun1: corpus frequency of hyponym
        """
        print(f"\n[Typicality] Computing GPT-2 typicality scores...")
        
        # Load GPT-2 if not loaded
        if self.gpt2_model is None:
            print("  Loading GPT-2...")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self.gpt2_model.eval()
        
        log_probs_noun2 = []
        log_probs_noun1 = []
        log_probs_conditional = []
        
        # Compute noun1 probability once (same for all rows)
        log_prob_noun1 = self._compute_gpt2_token_probability(self.noun1)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Typicality"):
            noun2 = row['predicted_hypernym']
            
            # P(noun2) - unconditional probability of hypernym
            log_prob_n2 = self._compute_gpt2_token_probability(noun2)
            log_probs_noun2.append(log_prob_n2)
            
            # P(noun1) - same for all rows
            log_probs_noun1.append(log_prob_noun1)
            
            # P(noun2 | "noun1 are a kind of") - conditional probability
            context = f"{self.noun1} are a kind of"
            log_prob_cond = self._compute_gpt2_conditional_probability(noun2, context)
            log_probs_conditional.append(log_prob_cond)
        
        df['log_prob_noun2'] = log_probs_noun2
        df['log_prob_noun1'] = log_probs_noun1
        df['log_prob_noun2_given_context'] = log_probs_conditional
        
        # Add word frequency using wordfreq
        print("  Adding word frequencies...")
        try:
            from wordfreq import word_frequency
            import math
            
            freq_noun2 = []
            for _, row in df.iterrows():
                text = row['predicted_hypernym']
                freq = word_frequency(text, 'en')
                log_freq = math.log(freq + 1e-10)
                freq_noun2.append(log_freq)
            
            df['log_wordfreq_noun2'] = freq_noun2
            
            # Word frequency for noun1 (same for all rows)
            freq_noun1 = word_frequency(self.noun1, 'en')
            df['log_wordfreq_noun1'] = math.log(freq_noun1 + 1e-10)
            
        except ImportError:
            print("  wordfreq not installed, skipping frequency scores")
        
        return df
    
    def balance_and_sample(self, df, total_samples=4000, seed=0, sampling_mode='random'):
        """Balance positives/negatives with configurable sampling strategy
        
        Args:
            df: DataFrame with gpt4_ground_truth column
            total_samples: Target number of balanced samples
            seed: Random seed for reproducibility (default 0)
            sampling_mode: 'random' (preserves distribution) or 'quantile' (equal from each log-prob quantile)
        
        Returns:
            Balanced DataFrame, or original if balancing impossible
        """
        print(f"\n[Balance] Creating balanced dataset with {total_samples} samples (mode={sampling_mode})...")
        
        np.random.seed(seed)
        random.seed(seed)
        
        positives = df[df['gpt4_ground_truth'] == 'Yes'].copy()
        negatives = df[df['gpt4_ground_truth'] == 'No'].copy()
        
        n_pos = len(positives)
        n_neg = len(negatives)
        print(f"  Raw counts: {n_pos} positives, {n_neg} negatives")
        
        n_per_class = total_samples // 2
        
        # Check if balancing is possible
        min_class_size = min(n_pos, n_neg)
        if min_class_size == 0:
            print(f"  WARNING: Cannot balance - one class has 0 samples!")
            print(f"           Positives: {n_pos}, Negatives: {n_neg}")
            print(f"           Returning unbalanced dataset.")
            return df
        
        if min_class_size < n_per_class:
            print(f"  WARNING: Cannot achieve target balance of {n_per_class} per class!")
            print(f"           Minority class has only {min_class_size} samples.")
            print(f"           Will use {min_class_size} per class instead ({min_class_size * 2} total).")
            n_per_class = min_class_size
        
        def random_sample(df_class, n_samples):
            """Simple random sampling - preserves original distribution"""
            if len(df_class) <= n_samples:
                return df_class
            return df_class.sample(n=n_samples, random_state=seed)
        
        def quantile_sample(df_class, n_samples):
            """Quantile-stratified sampling - equal representation from each log-prob quantile"""
            if len(df_class) <= n_samples:
                return df_class
            
            # Create log_prob bins
            df_class = df_class.copy()
            n_bins = min(10, max(1, len(df_class) // 10))
            if n_bins > 1:
                df_class['log_prob_bin'] = pd.qcut(df_class['log_prob'], q=n_bins, 
                                                    labels=False, duplicates='drop')
            else:
                df_class['log_prob_bin'] = 0
            
            # Sample proportionally from each bin
            n_unique_bins = df_class['log_prob_bin'].nunique()
            samples_per_bin = n_samples // n_unique_bins if n_unique_bins > 0 else n_samples
            sampled = []
            
            for bin_id in df_class['log_prob_bin'].unique():
                bin_data = df_class[df_class['log_prob_bin'] == bin_id]
                n_sample = min(len(bin_data), samples_per_bin + 10)  # +10 for buffer
                sampled.append(bin_data.sample(n=n_sample, random_state=seed))
            
            result = pd.concat(sampled, ignore_index=True)
            
            # Trim or pad to exact size
            if len(result) > n_samples:
                result = result.sample(n=n_samples, random_state=seed)
            elif len(result) < n_samples:
                # Sample more from full dataset
                remaining = n_samples - len(result)
                available = df_class[~df_class.index.isin(result.index)]
                if len(available) > 0:
                    extra = available.sample(n=min(remaining, len(available)), random_state=seed)
                    result = pd.concat([result, extra], ignore_index=True)
            
            return result.drop(columns=['log_prob_bin'], errors='ignore')
        
        # Choose sampling function based on mode
        if sampling_mode == 'random':
            sample_fn = random_sample
        elif sampling_mode == 'quantile':
            sample_fn = quantile_sample
        else:
            print(f"  WARNING: Unknown sampling_mode '{sampling_mode}', defaulting to 'random'")
            sample_fn = random_sample
        
        # Priority strategies: always include ALL samples from these
        priority_strategies = {'existing_hypernyms', 'beam_search'}
        
        def sample_with_priority(df_class, n_samples):
            """Sample while ensuring all priority strategy samples are included"""
            # Separate priority and non-priority samples
            priority = df_class[df_class['strategy'].isin(priority_strategies)]
            non_priority = df_class[~df_class['strategy'].isin(priority_strategies)]
            
            n_priority = len(priority)
            if n_priority > 0:
                print(f"    Including all {n_priority} samples from priority strategies")
            
            # If priority samples already exceed target, return all of them
            if n_priority >= n_samples:
                return priority
            
            # Otherwise, sample from non-priority to fill the rest
            n_needed = n_samples - n_priority
            sampled_non_priority = sample_fn(non_priority, n_needed)
            
            return pd.concat([priority, sampled_non_priority], ignore_index=True)
        
        sampled_pos = sample_with_priority(positives, n_per_class)
        sampled_neg = sample_with_priority(negatives, n_per_class)
        
        balanced = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        final_pos = len(balanced[balanced['gpt4_ground_truth']=='Yes'])
        final_neg = len(balanced[balanced['gpt4_ground_truth']=='No'])
        print(f"  Final balanced: {len(balanced)} samples")
        print(f"    Positives: {final_pos}")
        print(f"    Negatives: {final_neg}")
        
        if final_pos != final_neg:
            print(f"  WARNING: Dataset is not perfectly balanced ({final_pos} vs {final_neg})")
        
        return balanced
    
    def create_train_test_split(self, df, test_size=500, seed=0):
        """Create balanced train/test split
        
        Args:
            df: DataFrame with gpt4_ground_truth column
            test_size: Target number of test samples
            seed: Random seed for reproducibility (default 0)
        
        Returns:
            (train_df, test_df) tuple
        """
        print(f"\n[Split] Creating train/test split (test_size={test_size}, seed={seed})...")
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Split each class separately to ensure balance
        positives = df[df['gpt4_ground_truth'] == 'Yes'].copy()
        negatives = df[df['gpt4_ground_truth'] == 'No'].copy()
        
        n_pos = len(positives)
        n_neg = len(negatives)
        
        # Check if balanced split is possible
        min_class_size = min(n_pos, n_neg)
        if min_class_size == 0:
            print(f"  WARNING: Cannot create balanced split - one class has 0 samples!")
            print(f"           Positives: {n_pos}, Negatives: {n_neg}")
            # Fall back to random split
            df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            test_df = df_shuffled.iloc[:test_size]
            train_df = df_shuffled.iloc[test_size:]
            return train_df, test_df
        
        # Total balanced data available (use min class size for balance)
        total_balanced = min_class_size * 2
        
        # If not enough data, set test to 1/4 of total (rest for train)
        if test_size > total_balanced // 4:
            old_test_size = test_size
            test_size = total_balanced // 4
            # Make it even for balance
            test_size = (test_size // 2) * 2
            print(f"  INFO: Not enough data for test_size={old_test_size}.")
            print(f"        Total balanced samples: {total_balanced}")
            print(f"        Adjusting test_size to {test_size} (1/4 of data)")
        
        n_test_per_class = test_size // 2
        
        if min_class_size < n_test_per_class:
            print(f"  WARNING: Cannot achieve target test balance of {n_test_per_class} per class!")
            print(f"           Minority class has only {min_class_size} samples total.")
            # Adjust test size to use at most half of minority class
            n_test_per_class = min(n_test_per_class, min_class_size // 2)
            print(f"           Adjusting to {n_test_per_class} test samples per class.")
        
        # Shuffle both classes
        positives = positives.sample(frac=1, random_state=seed).reset_index(drop=True)
        negatives = negatives.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Split
        test_pos = positives.iloc[:n_test_per_class]
        train_pos = positives.iloc[n_test_per_class:]
        
        test_neg = negatives.iloc[:n_test_per_class]
        train_neg = negatives.iloc[n_test_per_class:]
        
        train_df = pd.concat([train_pos, train_neg], ignore_index=True)
        test_df = pd.concat([test_pos, test_neg], ignore_index=True)
        
        # Final shuffle
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        train_pos_count = len(train_df[train_df['gpt4_ground_truth']=='Yes'])
        train_neg_count = len(train_df[train_df['gpt4_ground_truth']=='No'])
        test_pos_count = len(test_df[test_df['gpt4_ground_truth']=='Yes'])
        test_neg_count = len(test_df[test_df['gpt4_ground_truth']=='No'])
        
        print(f"  Train: {len(train_df)} ({train_pos_count} pos, {train_neg_count} neg)")
        print(f"  Test: {len(test_df)} ({test_pos_count} pos, {test_neg_count} neg)")
        
        # Warn if not balanced
        if train_pos_count != train_neg_count:
            print(f"  WARNING: Train set is not balanced ({train_pos_count} pos vs {train_neg_count} neg)")
        if test_pos_count != test_neg_count:
            print(f"  WARNING: Test set is not balanced ({test_pos_count} pos vs {test_neg_count} neg)")
        
        return train_df, test_df
    
    def run_pipeline(self, total_samples=4000, test_size=500, output_dir="../data", 
                     save_garbage=False, sampling_mode='random', seed=0,
                     top_k=5000, num_beams=600, num_suggestions=400, num_gpt4_calls=1):
        """Run the full pipeline with checkpointing for resume capability
        
        Args:
            total_samples: Target number of balanced samples
            test_size: Number of samples for test set
            output_dir: Directory for output files
            save_garbage: If True, save filtered items to a garbage CSV file
            sampling_mode: 'random' or 'quantile' for balancing step
            seed: Random seed for reproducibility
            top_k: Top-k for 1-token sampling
            num_beams: Number of beams for beam search
            num_suggestions: Number of GPT-4 suggestions per call (for both pos and neg)
            num_gpt4_calls: Number of GPT-4 calls to make (results aggregated for diversity)
        """
        print(f"\n{'='*60}")
        print(f"RUNNING FULL PIPELINE FOR: {self.noun1}")
        print(f"Model: {self.model_name}")
        print(f"Sampling: top_k={top_k}, num_beams={num_beams}, num_suggestions={num_suggestions}, gpt4_calls={num_gpt4_calls}")
        print(f"{'='*60}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Checkpoint files (include model name to avoid conflicts)
        checkpoint_dir = output_dir / f".checkpoints_{self.noun1}_{self.model_name_safe}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        ckpt_completions = checkpoint_dir / "step1_completions.csv"
        ckpt_labeled = checkpoint_dir / "step2_labeled.csv"
        ckpt_typicality = checkpoint_dir / "step3_typicality.csv"
        garbage_file = output_dir / f"hypernym_{self.noun1}_{self.model_name_safe}_garbage.csv"
        
        # Step 1: Sample completions from all strategies
        if ckpt_completions.exists():
            print(f"\n[RESUME] Loading completions from checkpoint...")
            df = pd.read_csv(ckpt_completions)
            print(f"  Loaded {len(df)} completions")
        else:
            all_results = []
            all_garbage = []
            
            if save_garbage:
                results_1t, garbage_1t = self.sample_1token(top_k=top_k, collect_garbage=True)
                all_results.extend(results_1t)
                all_garbage.extend(garbage_1t)
                
                results_2t, garbage_2t = self.sample_2token(top_k1=200, top_k2=50, collect_garbage=True)
                all_results.extend(results_2t)
                all_garbage.extend(garbage_2t)
            else:
                all_results.extend(self.sample_1token(top_k=top_k))
                all_results.extend(self.sample_2token(top_k1=200, top_k2=50))
            
            all_results.extend(self.sample_beam_search(num_beams=num_beams))
            all_results.extend(self.sample_gpt4_positive(num_suggestions=num_suggestions, num_calls=num_gpt4_calls))
            all_results.extend(self.sample_gpt4_negative(num_suggestions=num_suggestions, num_calls=num_gpt4_calls))
            all_results.extend(self.sample_existing_hypernyms())
            
            # Save garbage if requested
            if save_garbage and all_garbage:
                garbage_df = pd.DataFrame(all_garbage)
                garbage_df = garbage_df.sort_values('log_prob', ascending=False)
                garbage_df.to_csv(garbage_file, index=False)
                print(f"\n[Garbage] Saved {len(garbage_df)} filtered items to {garbage_file}")
            
            # Combine and deduplicate (keep highest log_prob for each hypernym)
            df = pd.DataFrame(all_results)
            df = df.sort_values('log_prob', ascending=False)
            df = df.drop_duplicates(subset=['predicted_hypernym'], keep='first')
            df = df.reset_index(drop=True)
            print(f"\n[Combined] {len(df)} unique completions from all strategies")
            
            # Save checkpoint
            df.to_csv(ckpt_completions, index=False)
            print(f"  Checkpoint saved: {ckpt_completions}")
        
        # Step 2: Get GPT-4 labels (with incremental saving)
        if ckpt_labeled.exists():
            print(f"\n[RESUME] Loading labeled data from checkpoint...")
            df = pd.read_csv(ckpt_labeled)
            # Check if labeling is complete
            needs_label = df['gpt4_ground_truth'].isna() | (df['gpt4_ground_truth'] == '')
            if needs_label.any():
                print(f"  Resuming: {needs_label.sum()} samples still need labeling...")
                df = self.add_gpt4_labels(df, checkpoint_file=ckpt_labeled)
            else:
                print(f"  Loaded {len(df)} fully labeled samples")
        else:
            df = self.add_gpt4_labels(df, checkpoint_file=ckpt_labeled)
        
        # Filter unknowns
        df = df[df['gpt4_ground_truth'].isin(['Yes', 'No'])]
        print(f"  After filtering unknowns: {len(df)} samples")
        
        # Save final labeled checkpoint
        df.to_csv(ckpt_labeled, index=False)
        print(f"  Checkpoint saved: {ckpt_labeled}")
        
        # Step 3: Add typicality scores
        if ckpt_typicality.exists():
            print(f"\n[RESUME] Loading typicality data from checkpoint...")
            df = pd.read_csv(ckpt_typicality)
            print(f"  Loaded {len(df)} samples with typicality")
        else:
            df = self.add_typicality_scores(df)
            
            # Save checkpoint
            df.to_csv(ckpt_typicality, index=False)
            print(f"  Checkpoint saved: {ckpt_typicality}")
        
        # Step 4: Balance and sample
        df_balanced = self.balance_and_sample(df, total_samples=total_samples, 
                                               seed=seed, sampling_mode=sampling_mode)
        
        # Step 5: Create train/test split
        train_df, test_df = self.create_train_test_split(df_balanced, test_size=test_size, seed=seed)
        
        # Step 6: Save final files (include model name in filename)
        # Sort by log_prob descending before saving
        train_df = train_df.sort_values('log_prob', ascending=False)
        test_df = test_df.sort_values('log_prob', ascending=False)
        
        train_file = output_dir / f"hypernym_{self.noun1}_{self.model_name_safe}_train.csv"
        test_file = output_dir / f"hypernym_{self.noun1}_{self.model_name_safe}_test.csv"
        full_file = output_dir / f"hypernym_{self.noun1}_{self.model_name_safe}_full.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        df.to_csv(full_file, index=False)  # Save full dataset too
        
        # Clean up checkpoints on success
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"\n  Cleaned up checkpoint directory")
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Saved to:")
        print(f"  {train_file}")
        print(f"  {test_file}")
        print(f"  {full_file} (all samples before balancing)")
        
        # Show log_prob distribution
        print(f"\nlog_prob distribution in balanced dataset:")
        print(df_balanced['log_prob'].describe())
        
        return train_df, test_df, df


def main():
    parser = argparse.ArgumentParser(description="Generate balanced hypernym dataset")
    parser.add_argument("--noun1", type=str, required=True, help="The noun to generate hypernyms for (e.g., 'cars', 'fruit')")
    parser.add_argument("--total", type=int, default=4000, help="Total samples in balanced dataset")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name")
    parser.add_argument("--save_garbage", action="store_true", 
                        help="Save filtered-out completions to a separate CSV file for inspection")
    parser.add_argument("--sampling_mode", type=str, default="random", choices=["random", "quantile"],
                        help="Balancing strategy: 'random' (preserves distribution) or 'quantile' (equal from each log-prob quantile)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    # Sampling parameters
    parser.add_argument("--top_k", type=int, default=5000, help="Top-k for 1-token sampling")
    parser.add_argument("--num_beams", type=int, default=1000, help="Number of beams for beam search")
    parser.add_argument("--num_suggestions", type=int, default=200, help="Number of GPT-4 suggestions per call (for both positive and negative)")
    parser.add_argument("--num_gpt4_calls", type=int, default=2, help="Number of GPT-4 calls to make (results aggregated for diversity)")
    
    args = parser.parse_args()
    
    generator = HypernymDatasetGenerator(
        noun1=args.noun1,
        model_name=args.model
    )
    
    generator.run_pipeline(
        total_samples=args.total,
        test_size=args.test_size,
        output_dir=args.output_dir,
        save_garbage=args.save_garbage,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
        top_k=args.top_k,
        num_beams=args.num_beams,
        num_suggestions=args.num_suggestions,
        num_gpt4_calls=args.num_gpt4_calls
    )


if __name__ == "__main__":
    main()

