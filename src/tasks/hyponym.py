"""
Hyponym task - the reverse of hypernym.

Given a hypernym (category), ask for a hyponym (specific example).
This flips the roles of noun1/noun2 from the hypernym task.

Example:
- Hypernym task: "dogs are a kind of ___" → "animal"
- Hyponym task: "A kind of animal is ___" → "dog"
"""

import os
import sys
from string import Template

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
import utils

# Named tuple for prompt-completion pairs
from collections import namedtuple
PromptCompletion = namedtuple('PromptCompletion', ['prompt', 'completion'])


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Load and split hypernym data (same data as hypernym task).
    
    The data structure is the same, but we'll flip the roles when making prompts.
    """
    L = utils.load_noun_pair_data()
    
    if split_type == 'hyper':
        L_train, L_test = utils.split_train_test_no_overlap(L, seed=seed)
    elif split_type == 'random':
        L_train, L_test = utils.split_train_test(L, seed=seed, subsample=False, num_train=3000)
    elif split_type == 'both':
        L_train, L_test = utils.split_train_test_no_overlap_both(L, seed=2)
    else:
        raise ValueError(f"Wrong value for split_type: {split_type}")
    
    return L_train, L_test


def make_prompt_hyponymy(item, style="generator", shots="zero", neg=False, gen_response=None, variation=0):
    """
    Make a prompt for hyponym task (flipped from hypernym).
    
    In hypernym: "dogs are a kind of ___" → expects "animal" (item.noun2)
    In hyponym:  "A kind of animal is ___" → expects "dog" (item.noun1)
    """
    
    # Convert variation to int if it's a numeric string
    if isinstance(variation, str) and variation.isdigit():
        variation = int(variation)

    # Generator templates - asking for a hyponym given a hypernym
    if variation == 0:
        gtemplate = "Complete the sentence: An example of a $hypernym is a"
    elif variation == 1:
        gtemplate = "A kind of $hypernym is a"
    elif variation == 2:
        gtemplate = "Name a type of $hypernym:"
    elif variation == 3:
        gtemplate = "Give me an example of a $hypernym:"
    else:
        raise ValueError(f"Unknown variation: {variation}")

    if style == "generator":
        if shots == "zero":
            if neg:
                if item.taxonomic == "no":
                    prompt = Template(
                        "Complete the sentence: A $hypernym is not a kind of"
                    ).substitute(hypernym=item.noun2, word=item.noun1)
                else:
                    prompt = Template(gtemplate).substitute(hypernym=item.noun2, word=item.noun1)
            else:
                prompt = Template(gtemplate).substitute(hypernym=item.noun2, word=item.noun1)
            
            # Completion is noun1 (the hyponym) with leading space
            completion = " " + item.noun1
            return PromptCompletion(prompt=prompt, completion=completion)
        
        elif shots == "few":
            # Few-shot examples for hyponym task
            few_shot_examples = [
                ("animal", "dog"),
                ("fruit", "apple"),
                ("vehicle", "car"),
                ("furniture", "chair"),
            ]
            
            prompt = "Complete the sentence by naming a specific example.\n\n"
            for hypernym, hyponym in few_shot_examples:
                prompt += f"An example of a {hypernym} is a {hyponym}\n"
            prompt += f"\nAn example of a {item.noun2} is a"
            
            completion = " " + item.noun1
            return PromptCompletion(prompt=prompt, completion=completion)
    
    elif style == "discriminator":
        # Discriminator: Is X a kind of Y?
        # For hyponym task: Is noun1 a kind of noun2?
        response = gen_response if gen_response else item.noun1
        
        if shots == "zero":
            prompt = f"Is a {response} a kind of {item.noun2}? Answer Yes or No.\nAnswer:"
        else:
            # Few-shot discriminator
            prompt = (
                "Answer whether the first word is a type/kind of the second word.\n\n"
                "Is a dog a kind of animal? Answer: Yes\n"
                "Is a car a kind of fruit? Answer: No\n"
                "Is an apple a kind of fruit? Answer: Yes\n"
                "Is a chair a kind of vehicle? Answer: No\n\n"
                f"Is a {response} a kind of {item.noun2}? Answer:"
            )
        
        # Completion is Yes or No based on taxonomic label
        label = item.taxonomic.strip().lower()
        completion = " Yes" if label == "yes" else " No"
        return PromptCompletion(prompt=prompt, completion=completion)
    
    else:
        raise ValueError(f"Unknown style: {style}")


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """
    Create prompt for generator or discriminator.
    """
    return make_prompt_hyponymy(
        item,
        style=style,
        shots=shots,
        neg=neg,
        gen_response=gen_response,
        variation=variation
    )


def get_completion(item):
    """
    Extract generator completion text from data item.
    
    For hyponym, this is noun1 (the hyponym) with leading space.
    FLIPPED from hypernym which returns noun2.
    """
    return " " + item.noun1


def get_label(item):
    """
    Extract binary label from data item.
    
    The label is stored in item.taxonomic ('yes' or 'no').
    Same as hypernym task.
    """
    return item.taxonomic.strip().lower()


# Register this task with the registry
register_task({
    'name': 'hyponym',
    'load_data': load_data,
    'make_prompt': make_prompt,
    'get_completion': get_completion,
    'get_label': get_label,
    
    # Match legacy batch sizes
    'batch_size': {'with_ref': 1, 'without_ref': 6},
    
    # Same split types as hypernym
    'supports_split_types': ['random', 'hyper', 'both'],
    
    'description': 'Hyponym task: given a hypernym, predict a hyponym (reverse of hypernym task)',
})

