"""
Utilities to:

- Load the hypernym dataset:
    L = load_noun_pair_data()

- Make a train/test split:
    L_train, L_test = split_train_test(L, seed=0, subsample=False, num_train=3000)

- Make a prompt to evaluate:
    # item is an element of L
    prompt = make_prompt_hypernymy(item, style="generator", shots="zero", neg=False)

- Make prompts and format them into a tokenized, padded, huggingface Dataset
    prompts_train, hf_train = make_and_format_data(make_prompt, L_train, tokenizer, style="discriminator", shots="few", neg=False, both="union")

- Load huggingface or peft model:
    model, tokenizer = load_model(peft_model_id, device)

Note: some of this code adapted from
https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy

"""

import re
import math
import json
import random
from collections import namedtuple
from string import Template
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import immutabledict
from find_disjoint_sets import partition_items_kernighan_lin
from task_registry import get_task

def write_data(filename, data):
    with open(filename, 'a') as fout:
        for sample in data:
            fout.write(json.dumps(sample))
            fout.write('\n')

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def filtering_hypernym(item, pos=True):
    if pos:
        return item.taxonomic == "yes"
    else:
        return item.taxonomic == "no"

def filtering_swords(item, pos=True):
    if pos:
        return item.synonym == "yes"
    else:
        return item.synonym == "no"

def filtering_triviaqa(item):
    if 'correct' in item:
        return item['correct'] == 'Yes'
    else:
        return True
def filtering_lambada(item): 
    if 'correct' in item:
        return item['correct'] == 'Yes'
    else:
        return True
def load_noun_pair_data():
    """
    Load the noun pair (hypernymy) data, with information including whether
    taxonomic relation exists ("taxonomic"), and high or low similarity ("sim")
    """
    with open("../data/ranks.txt", "r") as fd:
        x = fd.readlines()

    x = [i.strip().split("\t") for i in x]
    for i in x:
        i[4] = int(i[4])
    Item = namedtuple(
        "Item",
        ["noun1", "noun2", "taxonomic", "sim", "gen_rank", "yesgreater", "argmax"],
    )
    out = [Item(*i) for i in x]
    return out


def load_hypernym_noun_data(noun, v2=True):
    """
    Load a hypernym dataset for a specific noun with balanced train/test splits.
    
    Args:
        noun: The noun name (e.g., 'cars', 'fruit', 'dogs')
        v2: If True, load from fixed-hypernyms/ with grammar-corrected sentences
        
    Returns (L_train, L_test) with Item namedtuples matching hypernym format.
    
    Expected files (v2=False):
        ../data/hypernym_{noun}_train.csv
        ../data/hypernym_{noun}_test.csv
    
    Expected files (v2=True):
        ../data/fixed-hypernyms/hypernym_{noun}_google-gemma-2-2b_{train|test}-fixed.csv
    
    CSV columns:
        noun1, predicted_hypernym, gpt4_ground_truth (Yes/No), log_prob, ...
        (v2 also has: fixed_hypernym_generator, discriminator_sentence)
    """
    import pandas as pd
    from pathlib import Path
    
    if v2:
        # Load from fixed-hypernyms directory with grammar-corrected data
        train_file = Path(f"../data/fixed-hypernyms/hypernym_{noun}_google-gemma-2-2b_train-fixed.csv")
        test_file = Path(f"../data/fixed-hypernyms/hypernym_{noun}_google-gemma-2-2b_test-fixed.csv")
    else:
        # Try both naming patterns: with and without model name
        train_file = Path(f"../data/hypernym_{noun}_google-gemma-2-2b_train.csv")
        test_file = Path(f"../data/hypernym_{noun}_google-gemma-2-2b_test.csv")
        # Fallback to old naming if new doesn't exist
        if not train_file.exists():
            train_file = Path(f"../data/hypernym_{noun}_train.csv")
        if not test_file.exists():
            test_file = Path(f"../data/hypernym_{noun}_test.csv")
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}\n"
                                f"Run: python typicality/generate_hypernym_dataset.py --noun1 {noun}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}\n"
                                f"Run: python typicality/generate_hypernym_dataset.py --noun1 {noun}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    if v2:
        # v2 mode: include fixed_hypernym_generator and discriminator_sentence
        Item = namedtuple(
            "Item",
            ["noun1", "noun2", "taxonomic", "sim", "gen_rank", "yesgreater", "argmax",
             "fixed_hypernym_generator", "discriminator_sentence"],
        )
        
        def df_to_items(df):
            items = []
            for _, row in df.iterrows():
                taxonomic = "yes" if row['gpt4_ground_truth'] == 'Yes' else "no"
                item = Item(
                    noun1=row['noun1'],
                    noun2=row['fixed_hypernym_generator'],  # Use corrected hypernym as noun2
                    taxonomic=taxonomic,
                    sim="high",
                    gen_rank=0,
                    yesgreater="yes",
                    argmax="yes",
                    fixed_hypernym_generator=row['fixed_hypernym_generator'],
                    discriminator_sentence=row['discriminator_sentence'],
                )
                items.append(item)
            return items
    else:
        # Original mode
        Item = namedtuple(
            "Item",
            ["noun1", "noun2", "taxonomic", "sim", "gen_rank", "yesgreater", "argmax"],
        )
        
        def df_to_items(df):
            items = []
            for _, row in df.iterrows():
                taxonomic = "yes" if row['gpt4_ground_truth'] == 'Yes' else "no"
                item = Item(
                    noun1=row['noun1'],
                    noun2=row['predicted_hypernym'],
                    taxonomic=taxonomic,
                    sim="high",
                    gen_rank=0,
                    yesgreater="yes",
                    argmax="yes"
                )
                items.append(item)
            return items
    
    L_train = df_to_items(train_df)
    L_test = df_to_items(test_df)
    
    return L_train, L_test


def load_hypernym_car_data():
    """
    Load the hypernym-car dataset. Wrapper for load_hypernym_noun_data('cars').
    """
    return load_hypernym_noun_data('cars')


def load_lambada_data(seed=0, sample_negative=True): 
    if sample_negative:
        L_train = read_data('../data/dataset/lambada_pn_train.jsonl')
        L_test = read_data('../data/dataset/lambada_pn_test.jsonl')
    else:
        dataset =  load_dataset("EleutherAI/lambada_openai", "en", split="test")
        # NOTE: this is the same version Jennifer Hu used, see also
        # https://github.com/jennhu/lm-task-demands/blob/d28b94b9d83a9ad855734dae44e7582029fcc13e/src/metrics/lambada.py#L24
        L = []
        for i, example in enumerate(dataset):
            text = example["text"]
            # Get final word to be predicted (by splitting on whitespace).
            # NOTE: there's some debate about what the "true" Lambada task is:
            # https://github.com/EleutherAI/lm-evaluation-harness/issues/350
            splits = text.split(" ")
            prefix = " ".join(splits[:-1])
            final_word = splits[-1]
            item = {"context": prefix, "final_word": final_word}
            L.append(item)
        random.seed(seed)
        random.shuffle(L)
        L_train, L_test = L[:4153], L[4153:]
    return L_train, L_test

def load_triviaqa_data(seed=0, sample_negative=True):
    if sample_negative:
        L_train = read_data('../data/dataset/triviaqa_pn_train.jsonl')
        L_test = read_data('../data/dataset/triviaqa_pn_test.jsonl')
    else:
        # load data here
        L = load_dataset('lucadiliello/triviaqa') #TODO check if this is correct version.
        #USE SUBSET FOR NOW
        L_train =  L['train'].shuffle(seed=42).select(range(3000)) 
        L_test = L['validation'].shuffle(seed=42).select(range(1000))
    print("L_train:", len(L_train))
    print("L_test:", len(L_test))
    return L_train, L_test

def load_swords_data(seed=0):
    """Load the swords dataset, and makes positive and negative pairs from it."""
    #NOTE for now pick the top highest ranked item in list of substitutes as positive case

    with open("../data/swords-data-v1.1_test.json", "r") as fd:
        test = json.load(fd)

    with open("../data/swords-data-v1.1_dev.json", "r") as fd:
        dev = json.load(fd)

    Item = namedtuple(
        "Item",
        ["context", "target", "replacement", "synonym"],
    )


    #NOTE use their dev as my train to keep things sane.
    testset = []
    for item in test:
        #TODO make this change permanent in the file
        item['context'] = re.sub(r'\s+', ' ', item['context'])

        pos_replacement = item['substitutes'][0][0]
        neg_replacement = item['substitutes'][-1][0]

        context = item['context'].replace(item['target'], "*" + item['target'] + "*")

        testset.append(Item(context, item['target'], pos_replacement, 'yes') )
        testset.append(Item(context, item['target'], neg_replacement, 'no') )

    trainset = []
    for item in dev:
        item['context'] = re.sub(r'\s+', ' ', item['context'])

        pos_replacement = item['substitutes'][0][0]
        neg_replacement = item['substitutes'][-1][0]

        context = item['context'].replace(item['target'], "*" + item['target'] + "*")

        trainset.append(Item(context, item['target'], pos_replacement, 'yes') )
        trainset.append(Item(context, item['target'], neg_replacement, 'no') )

    #filter here
    #NOTE doing it this way we don't have a perfectly even balance (e.g., one yes and one no per example)
    # but close..
    testset = [i for ii,i in enumerate(testset) if
            (len(i.replacement.split(" "))<=3) and
            (i.replacement.split(" ")[0] not in ['the','be','a', 'in' 'yet', 'at', 'by', 'do', 'dont', 'we', 'and', 'even', 'to', 'with']) and
            i.replacement != i.target ]

    trainset = [i for ii,i in enumerate(trainset) if
            (len(i.replacement.split(" "))<=3) and
            (i.replacement.split(" ")[0] not in ['the','be','a', 'in' 'yet', 'at', 'by', 'do', 'dont', 'we', 'and', 'even', 'to', 'with']) and
            i.replacement != i.target ]

#    trainset = [i for i in trainset if
#            (len(i.replacement.split(" "))<=3) and
#            True]

    random.seed(seed)
    random.shuffle(trainset)
    random.shuffle(testset)
    return trainset, testset


def load_ifeval_data(seed=0):
    """Load the ifeval dataset, and makes positive and negative pairs from it."""
    dataset = read_data('../data/ifeval-data.jsonl')

    return split_train_test(dataset, seed=seed, subsample=False, num_train=math.floor(len(dataset) * 3 / 4))

def load_collie_data(seed=0):
    """Load the collie dataset, and makes positive and negative pairs from it."""
    dataset = read_data('../data/collie-word-containment.jsonl')

    return split_train_test(dataset, seed=seed, subsample=False, num_train=math.floor(len(dataset) * 3 / 4))


def make_prompt_lambada(item, style='generator', shots='zero', neg=False, gen_response = None):
    if style == "generator":

        #generator_prompt = 'Complete the story:$context '
        generator_prompt = 'What word is most likely to come next in the following text?\nText: $context '
        prompt = Template(generator_prompt).substitute(context=item['context'])
        #completion = " " + item.replacement
        completion = " " + item['final_word']
        #TODO should few-shot negation case have different example..??
        if shots == "few":
            examples = 'What word is most likely to come next in the following text?\n She gently takes him by his shoulders, forcing him to face her, and she adjusts the angle of his tie the way she might straighten a picture on the wall. "I\'m sure I don\'t need to tell you how important this gala is.\"\n\n\"You don\'t, but you will anyway.\n\n'
            #TODO make "neg" and regular version of this example??
            prompt = examples + prompt

    elif style == "discriminator":
        instruction =  ''
        examples = 'Is the word "anyway" the most likely word to come next in the following text?\nText: "She gently takes him by his shoulders, forcing him to face her, and she adjusts the angle of his tie the way she might straighten a picture on the wall. "I\'m sure I don\'t need to tell you how important this gala is."\n\n"You don\'t, but you will\"\n\nAnswer: Yes\n\n'

        template_string = 'Is the word "$final_word" the most likely word to come next in the following text?\nText: "$context"\n\nAnswer:'

        cur_final_word = gen_response if gen_response else item['final_word']

        if 'correct' in item:
            completion = " " + item['correct']
        else:
            completion = " Yes"
        if shots == 'zero':
            prompt = Template(
                instruction + template_string
                ).substitute(context=item['context'], final_word=cur_final_word)
            #completion = " " + item.synonym.capitalize(i)
            #NOTE only positives yere
            # completion = " Yes"

        #TODO more than two shots??
        if shots == "few":
            #example1 = "In the following sentence: 'Well, kid, what do you think? Remember, this is your quest.', can the word think be replaced by the word mind? Answer: No\n\n"
            #example2 = "In the following sentence: 'I thought as much. Now leave, before I call the rats on you.', can the word call be replaced by the word summon? Answer: Yes\n\n"
            prompt = Template(
                instruction + examples + template_string
                ).substitute(context=item['context'], final_word=cur_final_word)
            #completion = " " + item.synonym.capitalize()
            # completion = " Yes"
    else:
        raise ValueError("!?")

    prompt = prompt.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)



#TODO generalize better to DRY!
def make_prompt_triviaqa(item, with_context=False, style='generator', shots='zero', neg=False, gen_response=None):
    """ Only positive examples for now! Also, when multiple acceptable answers are available in the dataset,
    use the first one for now. """

    example_context = """[DOC] [TLE] brindisi | Italian music | Britannica.combrindisi | Italian music | Britannica.com [PAR] Italian music [PAR] THIS IS A DIRECTORY PAGE. Britannica does not currently have an article on this topic. [PAR] Learn about this topic in these articles: [PAR]   [PAR] in drinking song [PAR] ...in certain types of 19th-century opera and operetta, frequently involving not only a soloist but also a chorus joining in with choral repeats or refrains. In Italy the drinking song is known as brindisi (Italian: "toast"). In Giuseppe Verdi's operas drinking songs range from the cheerful "Libiamo" ("Let Us Drink") in La traviata (1853), to.."""

    example_question = "What kind of song is a Brindisi?"
    example_answer = "drinking song"

    if style=='generator':
        if with_context:
            example = "Context: " + example_context + "\n\nQuestion: " + example_question + "\n\nAnswer: " + example_answer + "\n\n"
        else:
            example = "Question: " + example_question + "\n\nAnswer: " + example_answer + "\n\n"

        query = "Question: " + item['question'] + "\n\nAnswer: "# + item['answer'] + "\n\n"
        completion = item['answers'][0]
        if with_context:
            query = "Context: "+ item['context'] + "\n\n" + query

        if shots =='zero':
            prompt = query
        else:
            prompt = example + query
    else: #discriminator
        if 'correct' in item:
            completion = item['correct']
        else:
            completion = 'Yes'
        example = "Is the correct answer to the question \"" + example_question + "\" given by \""+ example_answer + "\"? Answer Yes or No: " + completion + "\n\n"
        if with_context:
            example = "Context: " + example_context + "\n\n" + example + "\n\n"

        cur_answer = gen_response if gen_response else item['answers'][0]
        query = "Is the correct answer to the question \"" + item['question'] + "\" given by \""+ cur_answer + "\"? Answer Yes or No: "
        if with_context:
            query = "Context: "+ item['context']  + "\n\n" + query

        if shots == 'zero':
            prompt = query
        else:
            prompt = example + query
    
    prompt = prompt.strip()
    completion = " " + completion.strip().capitalize()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion", "answers"])
    return Pt(prompt, completion, item['answers'])


def make_prompt_swords(item, style="generator", shots="zero", neg=False, gen_response=None):
    """
    Make a prompt based on the item.
    """

    if neg and item.synonym=='no':
        negation_word = " not"
    else:
        negation_word = ""

    if style == "generator":
        generator_prompt = 'Notice the word "$target" used in the context: "$context". In this context, the word "$target" is$optional_negation synonymous with "'
        #generator_prompt = 'Notice the word "$target" used in the context: "$context". In this context, the word "$target" has roughly the same meaning as "'
        prompt = Template(generator_prompt).substitute(context=item.context, target=item.target, optional_negation=negation_word)
        #completion = " " + item.replacement
        completion = "" + item.replacement
        #TODO should few-shot negation case have different example..??
        if shots == "few":
            #examples = "In the following sentence: 'I thought as much. Now leave, before I call the rats on you.', the word call is a synonym of the word summon\n\n"
            #TODO make "neg" and regular version of this example??
            #examples = 'Notice the word "artists" used in the context: "Many painters, sculptors, and other *artists* were inspired by Duchamp.". In this context, the word "artists" is not synonymous with "character".\n\nNotice the word "happen" used in the context: "I could free Tasha. If I did, one of three things would *happen*. Most likely: she would be meat..." In this context, the word "happen" is synonymous with "transpire".\n\n'
            examples = 'Notice the word "happen" used in the context: "I could free Tasha. If I did, one of three things would *happen*. Most likely: she would be meat..." In this context, the word "happen" is synonymous with "transpire".\n\n'
            prompt = examples + prompt

    elif style == "discriminator":
        instruction =  'Determine whether the word in context can be replaced by another word or expression without changing the meaning of the sentence.\n\n'
        examples = 'Notice the word "artists" used in the context: "Many painters, sculptors, and other *artists* were inspired by Duchamp.". In this context, is "artists" synonymous with "character"? Answer: No\n\nNotice the word "happen" used in the context: "I could free Tasha. If I did, one of three things would *happen*. Most likely: she would be meat..." In this context, is "happen" synonymous with "transpire"? Answer: Yes\n\n'
        template_string = 'Notice the word "$target" used in the context: "$context". In this context, is "$target" synonymous with "$replacement"? Answer:'

        cur_replacement = gen_response if gen_response else item.replacement
        if shots == 'zero':
            prompt = Template(
                instruction + template_string
                ).substitute(context=item.context, target=item.target, replacement=cur_replacement)
            completion = " " + item.synonym.capitalize()

        #TODO more than two shots??
        if shots == "few":
            #example1 = "In the following sentence: 'Well, kid, what do you think? Remember, this is your quest.', can the word think be replaced by the word mind? Answer: No\n\n"
            #example2 = "In the following sentence: 'I thought as much. Now leave, before I call the rats on you.', can the word call be replaced by the word summon? Answer: Yes\n\n"
            prompt = Template(
                instruction + examples + template_string
                ).substitute(context=item.context, target=item.target, replacement=cur_replacement)
            completion = " " + item.synonym.capitalize()
    else:
        raise ValueError("!?")

    prompt = prompt.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)


def make_prompt_hypernymy(item, style="generator", shots="zero", neg=False, gen_response=None, variation=0):
    """
    Make a prompt based on the item.
    """
    
    # Convert variation to int if it's a numeric string, otherwise keep as is
    if isinstance(variation, str) and variation.isdigit():
        variation = int(variation)

    #TODO do this better later
    #variation=0 # 0 for original
    if variation==0:
        gtemplate = "Complete the sentence: $word are a kind of"
    elif variation==1:
        gtemplate = "$word are a kind of"
    elif variation==2:
        gtemplate = "I love $word and other"
    elif variation ==3:
        gtemplate = "Do you remember what our teacher used to tell us? She'd say that contrary to appearances, $word are actually"
        #"Do you remember what our teacher used to tell us? She'd say that contrary to appearances, $word are actually"
    elif variation=='kind-of':
        #v0-prompt gtemplate = "corgis are a kind of dog? Answer: Yes\n\n$word are a kind of"
        #v1 prompt gtemplate = "Do you think bees are furniture? Answer: No\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a fruit? Answer: No\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a"
        gtemplate = "Do you think sofas are furniture? Answer: Yes\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a vehicles? Answer: Yes\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a"
    else:
        raise ValueError("Wrong num")

    if style == "generator":
        if shots == "zero":
            if neg:
                if item.taxonomic == "no":
                    prompt = Template(
                        "Complete the sentence: $word are not a kind of"
                    ).substitute(word=item.noun1, hypernym=item.noun2)
                else:
                    prompt = Template(
                        gtemplate
                        #""
                        #"Complete the sentence: $word have a tendency to"
                        #"Do you remember what our teacher used to tell us? She'd say that contrary to appearances, $word are actually"
                        #"I love $word and other"
                        #"$word are a kind of"
                        #"Complete the sentence: $word are a kind of"
                        #NOTE
                    ).substitute(word=item.noun1, hypernym=item.noun2)
            else:
                prompt = Template(
                        gtemplate
                    #"Complete the sentence: $word are a kind of"
                        #"$word are a kind of"
                        #"I love $word and other"
                        #"Do you remember what our teacher used to tell us? She'd say that contrary to appearances, $word are actually"
                        #"Do you remember what our teacher used to tell us? She'd say that contrary to appearances, $word are actually"
                        #"Complete the sentence: $word have a tendency to"
                ).substitute(word=item.noun1, hypernym=item.noun2)
            completion = " " + item.noun2
        else:
            if neg:
                if item.taxonomic == "no":
                    prompt = Template(
                        "Complete the sentence: bees are not a kind of furniture\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are not a kind of fruit\n\nComplete the sentence: $word are not a kind of"
                    ).substitute(word=item.noun1)
                else:
                    prompt = Template(
                        "Complete the sentence: bees are a kind of insect\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are a kind of bird\n\nComplete the sentence: $word are a kind of"
                    ).substitute(word=item.noun1)
            else:
                prompt = Template(
                    "Complete the sentence: bees are a kind of insect\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are a kind of bird\n\nComplete the sentence: $word are a kind of"
                ).substitute(word=item.noun1)
            completion = " " + item.noun2
    elif style == "discriminator":
        cur_hypernym = gen_response if gen_response else item.noun2
        if shots == "zero":
#            prompt = Template("Do you think $word are a $hypernym? Answer:").substitute(
#                word=item.noun1, hypernym=cur_hypernym
#            )
#           NOTE
            prompt = Template("Is it the case that $word are a $hypernym? Answer:").substitute(
                word=item.noun1, hypernym=cur_hypernym
            )
#            prompt = Template("In your view, are $word a $hypernym? Answer:").substitute(
#                word=item.noun1, hypernym=cur_hypernym
#            )

            completion = " " + item.taxonomic.capitalize()
        else:
            if variation==0:
                prompt = Template(
                "Do you think bees are furniture? Answer: No\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a fruit? Answer: No\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a $hypernym? Answer:"
                ).substitute(word=item.noun1, hypernym=cur_hypernym)
            elif variation==1:
                prompt = Template(
                        #NOTE
                    "Is it the case that bees are furniture? Answer: No\n\nIs it the case that corgis are dogs? Answer: Yes\n\nIs it the case that trucks are a fruit? Answer: No\n\nIs it the case that robins are birds? Answer: Yes\n\nIs it the case that $word are a $hypernym? Answer:"
               ).substitute(word=item.noun1, hypernym=cur_hypernym)
            elif variation==2:
                prompt = Template(
                        #NOTE
                    "In your view, are bees furniture? Answer: No\n\nIn your view, are corgis dogs? Answer: Yes\n\nIn your view, are trucks a fruit? Answer: No\n\nIn your view, are robins birds? Answer: Yes\n\nIn your view, are $word a $hypernym? Answer:"
                ).substitute(word=item.noun1, hypernym=cur_hypernym)
            elif variation==3:
                prompt = Template(
                    "Deep down in your bones, do you believe that bees are furniture? Answer: No\n\nDeep down in your bones, do you believe that corgis are dogs? Answer: Yes\n\nDeep down in your bones, do you believe that trucks are a fruit? Answer: No\n\nDeep down in your bones, do you believe that robins are birds? Answer: Yes\n\nDeep down in your bones, do you believe that $word are a $hypernym? Answer:"
               ).substitute(word=item.noun1, hypernym=cur_hypernym)
            elif variation==4:
                prompt = Template(
                    "Deep down in your bones, do you believe that bees are furniture? Answer: No\n\nDeep down in your bones, do you believe that corgis are dogs? Answer: Yes\n\nDeep down in your bones, do you believe that trucks are a fruit? Answer: No\n\nDeep down in your bones, do you believe that robins are birds? Answer: Yes\n\nDeep down in your bones, do you believe that $word are a $hypernym? Answer:"
                ).substitute(word=item.noun1, hypernym=cur_hypernym)
            elif variation=='kind-of':
                #v0- prompt = Template("corgis are a kind of dog? Answer: Yes\n\n$word are a kind of $hypernym? Answer:").substitute(word=item.noun1, hypernym=cur_hypernym)
                #v1- prompt
                #prompt = Template(
                #"Do you think bees are furniture? Answer: No\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a fruit? Answer: No\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a $hypernym? Answer:"
                #).substitute(word=item.noun1, hypernym=cur_hypernym)
                prompt = Template(
                "Do you think sofas are furniture? Answer: Yes\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a vehicles? Answer: Yes\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a $hypernym? Answer:"
                ).substitute(word=item.noun1, hypernym=cur_hypernym)
            else:
                raise ValueError("Wrong num!")

            completion = " " + item.taxonomic.capitalize()
    else:
        raise ValueError("!?")

    prompt = prompt.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)


def make_prompt_hypernymy_v2(item, style="generator", shots="zero", neg=False, gen_response=None, variation=0):
    """
    Make a prompt for v2 hypernym data (grammar-corrected).
    
    Uses fixed_hypernym_generator for completion and discriminator_sentence for discriminator prompts.
    """
    # Convert variation to int if it's a numeric string
    if isinstance(variation, str) and variation.isdigit():
        variation = int(variation)
    
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    
    if style == "generator":
        # Use the same gtemplate as original variation=0
        gtemplate = "Complete the sentence: $word are a kind of"
        prompt = Template(gtemplate).substitute(word=item.noun1)
        # Use fixed_hypernym_generator as completion
        completion = " " + item.fixed_hypernym_generator
        
    elif style == "discriminator":
        if variation != 0:
            raise NotImplementedError(f"v2 discriminator only supports variation=0, got variation={variation}")
        
        # Few-shot prefix + discriminator_sentence from the CSV
        # v1 few-shot (animate bias - Yes for animals):
        # few_shot_prefix = (
        #     "Do you think bees are furniture? Answer: No\n\n"
        #     "Do you think corgis are dogs? Answer: Yes\n\n"
        #     "Do you think trucks are a fruit? Answer: No\n\n"
        #     "Do you think robins are birds? Answer: Yes\n\n"
        # )
        # v2 few-shot (balanced - Yes for both animate and inanimate):
        few_shot_prefix = (
            "Do you think bees are dolphins? Answer: No\n\n"
            "Do you think corgis are dogs? Answer: Yes\n\n"
            "Do you think robins are fruit? Answer: No\n\n"
            "Do you think trucks are vehicles? Answer: Yes\n\n"
        )
        # discriminator_sentence already contains the question (e.g., "Do you think bananas are a food?")
        prompt = few_shot_prefix + item.discriminator_sentence + " Answer:"
        completion = " " + item.taxonomic.capitalize()
    else:
        raise ValueError(f"Unknown style: {style}")
    
    prompt = prompt.strip()
    return Pt(prompt, completion)


# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
    })

def map_ifeval_instruction(instruction, args):
    match instruction:
        case "keywords:existence": 
            keywords = args['keywords']
            keywords_str = ', '.join(f"'{word}'" for word in keywords)
            return f"contains the word(s) {keywords_str}"
        
        case "keywords:frequency": 
            return f"contains the word '{args['keyword']}' {args['relation']} {args['frequency']} times"
        
        case "keywords:forbidden_words": 
            forbidden_words = args['forbidden_words']
            forbidden_words_str = ', '.join(f"'{word}'" for word in forbidden_words)
            return f"does not contain the word(s) {forbidden_words_str}"
        
        case "keywords:letter_frequency": 
            return f"contains the letter '{args['letter']}' {args['let_relation']} {args['let_frequency']} times"
        
        case "language:response_language":
            return f"written in {LANGUAGE_CODES.get(args['language'], args['language'])}"
        
        case "length_constraints:number_sentences":
            return f"contains {args['relation']} {args['num_sentences']} sentences"
        
        case "length_constraints:number_paragraphs":
            return f"contains {args['num_paragraphs']} paragraphs"
        
        case "length_constraints:number_words":
            return f"contains {args['relation']} {args['num_words']} words"
        
        case "length_constraints:nth_paragraph_first_word": 
            return f"contains {args['num_paragraphs']} paragraphs separted by two new lines and the first word of paragraph {args['nth_paragraph']} is '{args['first_word']}'"
        
        case "detectable_content:number_placeholders": 
            return f"contains at least {args['num_placeholders']} placeholders denoted by square brackets, such as [address]"
        
        case "detectable_content:postscript": 
            return f"ends with a postscript denoted by '{args['postscript_marker']}'"
        
        case "detectable_format:number_bullet_lists":
            return f"contains {args['num_bullets']} bullet points denoted by * in markdown"
        
        case "detectable_format:constrained_response": 
            return "is exactly one of the following: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'"
        
        case "detectable_format:number_highlighted_sections": 
            return f"contains at least {args['num_highlights']} higlighted sections denoted by *highlighted section* in markdown"
        
        case "detectable_format:multiple_sections": 
            return f"has {args['num_sections']} sections with each section beginning with '{args['section_spliter']} X' with X being the section number"
        
        case "detectable_format:json_format": 
            return "in a valid JSON format"
        
        case "detectable_format:title": 
            return "must contain a title wrappec in double angular brackets i.e. <<title>>"
        
        case "combination:two_responses": 
            return "has two responses separated by 6 asterisks ******"
        
        case "combination:repeat_prompt":
            return f"starts with {args['prompt_to_repeat']}"
        
        case "startend:end_checker": 
            return f"ends with {args['end_phrase']}"
        
        case "change_case:capital_word_frequency":
            return f"contains {args['capital_relation']} {args['capital_frequency']} words in all capital letters"
        
        case "change_case:english_capital": 
            return "is written in English in all capital letters"
        
        case "change_case:english_lowercase": 
            return "is written in English in all lower case letters"
        
        case "punctuation:no_comma": 
            return "does not contain any commas"
        
        case "startend:quotation": 
            return "wrapped in double quotation marks"


def make_prompt_ifeval(item, style="generator", shots="zero", gen_response=None, neg = False):
    """
    Make a prompt based on the item.
    """

    if neg:
        raise NotImplementedError("negation not implemented for ifeval yet")

    if style == "generator":
        prompt = item['prompt']

        if shots != "zero":
            raise NotImplementedError("TODO") # add example to prompt
        
        completion = " " + item['response']

    elif style == "discriminator":
        curr_response = gen_response if gen_response else item['response']
        # todo improve prompt to ask about specific instructions. (create a mapping from the instrustions in the instruction registry to text).
        
        # original prompt
        prompt = Template(
            "You are a strict binary validator.\n\n"
            "You will be given a prompt and a response. Determine if the response follows ALL instructions in the prompt exactly.\n\n"
            "Rules:\n"
            "- Every constraint must be satisfied perfectly.\n"
            "- If ANY constraint is violated, output NO.\n"
            "- Do not give partial credit.\n"
            "- Do not infer or assume missing requirements.\n"
            "- If you are uncertain, output NO.\n\n"
            "Prompt: $prompt\n\n"
            "Response: $response\n\n"
            "Output exactly one word: YES or NO."
        ).substitute(prompt=item['prompt'], response=curr_response)
        
        # properties = [map_ifeval_instruction(instr, args) for instr, args in zip(item['instruction_id_list'], item['kwargs'])]
        # properties_str = "\n".join(f"- {prop}" for prop in properties)

        # text first
        # prompt = Template(
        #         "$text \n\nDoes the above text have the following properties (yes/no)?\n$properties_str"
        #         ).substitute(text=curr_response, properties_str=properties_str)

        # properties first
        # prompt = Template(
        #         "You will be given text, and will have to determine whether it has all of the following properties:\n$properties_str \n\n Text:\n$text \n\n Does the text have all of the properties? (Yes/No)"
        #         ).substitute(text=curr_response, properties_str=properties_str)
        
        completion = " " + item['correct']

        if shots != "zero":
            raise NotImplementedError("TODO") # add example to prompt
    else:
        raise ValueError("!?")
    
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt.strip(), completion)
    
def make_prompt_collie(item, style="generator", shots="zero", gen_response=None, neg = False): 

    if neg:
        raise NotImplementedError("negation not implemented for collie yet")

    if style == "generator":
        if item["satisfies_constraint"]:
            prompt = item['prompt']
        else:
            prompt = item['prompt'] # negative examples should prob have a different prompt?

        if shots != "zero":
            raise NotImplementedError("TODO") # add example to prompt
        
        completion = " " + item['generated']

    elif style == "discriminator":
        curr_response = gen_response if gen_response else item['generated']
        collie_restriction = item['prompt'][16:] #remove "Please generate "
        prompt = Template(
                "You will be given a sentence and a list of words. Determine if the following sentence contains all of these words exactly. Answer yes or no with no explanation. $response. Is this $collie_restriction?"
                ).substitute(prompt=item['prompt'], response=curr_response, collie_restriction=collie_restriction)

        if shots != "zero":
            example = "You will be given a sentence and a list of words. Determine if the sentence contains all of these words exactly. Answer yes or no with no explanation. The cat sat on the mat. Is this a sentence containing the word 'on', 'cat'? Yes. The overcast sky loomed over the city. Is this a sentence containing the word 'blue', 'loom'? No. "
            prompt = example + prompt

        completion = " " + ("Yes" if item['satisfies_constraint'] else "No")

    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt.strip(), completion)

def split_train_test(L, seed=0, subsample=False, num_train=3000):
    """
    After loading the data into a list L, use this to shuffle, subsample (optional),
    and split into train and test sets.
    """
    random.seed(seed)
    random.shuffle(L)

    if subsample:
        L = L[9::10]
        if num_train > len(L):
            num_train = math.floor(len(L) * 3 / 4)
            print("Reset num_train to ", num_train)

    L_train = L[:num_train]
    L_test = L[num_train:]
    return L_train, L_test


def split_train_test_no_overlap(L, seed=0):
    #TODO later flag for option no overlap in hyper, in hypo, or both
    test_concepts_hyper = ['jewelry',
                           'home decor',
                           'vehicle',
                           'musical instrument',
                           'tool',
                           'container',
                           'auto part',
                           'kitchen equipment',
                           'kitchen tool',
                           'garden tool']
    random.seed(seed)
    random.shuffle(L)
    L_train = [i for i in L if i.noun2 not in test_concepts_hyper]
    L_test = [i for i in L if i.noun2 in test_concepts_hyper]
    return L_train, L_test

def f1(x): return [i.noun1 for i in x]
def f2(x): return [i.noun2 for i in x]

def split_train_test_no_overlap_both(L, seed=2):
    random.seed(seed)
    random.shuffle(L)
    c1, c2, exc = partition_items_kernighan_lin(L)
    assert len(c1) == 419
    assert len(c2) == 2676 # train
    assert len(set(f1(c1)).intersection(f1(c2)))==0
    assert len(set(f2(c1)).intersection(f2(c2)))==0
    L_train = c2
    L_test = c1
    return L_train, L_test


def make_and_format_data(
    make_prompt,
    L,
    tokenizer,
    style="discriminator",
    shots="few",
    neg=False,
    both="union",
    instruction_masking=True,
    filtering=None,
    is_chat = False,
):
    """
    Make prompts and completions, and tokenize and pad into a HF dataset.

    Note on `both`:
    - combines generator and discriminator, this ignores other parameters
    `style`, `shots` and `neg`
    """
    if filtering:
        print("Original L length: ", len(L))
        L_filter = list(filter(filtering, L))
        print("Filtered L to length: ", len(L_filter))
    else:
        L_filter = L

    if style is None or both == 'union':
        items_disc = [make_prompt(i, style='discriminator', shots=shots) for i in L]
        items_gen = [make_prompt(i, style='generator', shots=shots) for i in L_filter]
        items = items_disc + items_gen
        random.shuffle(items)



    elif both == "joint":
        items1 = [
            make_prompt(i, style="discriminator", shots="zero", neg=False) for i in L
        ]
        items2 = [make_prompt(i, style="generator", shots="zero", neg=True) for i in L]

        discfirst = [
            namedtuple("PromptCompletion", ["prompt", "completion"])(
                items1[ii].prompt + items1[ii].completion + "\n\n" + items2[ii].prompt,
                items2[ii].completion,
            )
            for ii in range(len(items1))
        ]

        genfirst = [
            namedtuple("PromptCompletion", ["prompt", "completion"])(
                items2[ii].prompt + items2[ii].completion + "\n\n" + items1[ii].prompt,
                items1[ii].completion,
            )
            for ii in range(len(items1))
        ]

        items = discfirst + genfirst
        random.shuffle(items)

    else:
        items = [make_prompt(i, style=style, shots=shots, neg=neg) for i in L_filter]
    
    print("EXAMPLE items: ")
    print("prefix: ", items[0].prompt)
    print("completion: ", items[0].completion)

    instructions = []
    completions = []
    if not is_chat:
        prompt_completion_data = []
        for ii in range(len(items)):
            prompt_completion_data.append({"prompt": items[ii].prompt.strip(), "completion": items[ii].completion})
            instruction = tokenizer(items[ii].prompt)["input_ids"]
            output = tokenizer(items[ii].completion, add_special_tokens=False)["input_ids"]
            instructions.append(instruction)
            completions.append(output)
    else:
        prompt_completion_data = []
        for ii in range(len(items)):
            prompt_completion_data.append(
                {"messages": [{"role": "system", "content": "You are a helpful assistant."}, 
                              {"role": "user", "content": items[ii].prompt.strip()}, 
                              {"role": "assistant", "content": items[ii].completion.strip()}]}
            )
            instruction = tokenizer(items[ii].prompt)["input_ids"]
            output = tokenizer(items[ii].completion, add_special_tokens=False)["input_ids"]
            instructions.append(instruction)
            completions.append(output)

    inputs = [instructions[ii] + completions[ii] for ii in range(len(completions))]

    # NOTE: The huggingface Training takes care of shifting tokens by 1.
    if instruction_masking:
        outputs = [
            [-100] * len(instructions[ii]) + completions[ii]
            for ii in range(len(completions))
        ]
    else:
        outputs = inputs

    padded_inputs = tokenizer.pad(
        {"input_ids": inputs}, padding=True, return_tensors="pt"
    )
    padded_outputs = tokenizer.pad(
        {"input_ids": outputs}, padding=True, return_tensors="pt"
    )

    hf_dataset = Dataset.from_dict(
        {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_outputs["input_ids"],
        }
    )

    return items, hf_dataset, Dataset.from_list(prompt_completion_data)


def load_model(peft_model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_id, torch_dtype=torch.float16
    )
    model.to(device)
    return model, tokenizer


def get_model_input_device(model, fallback_device='cuda'):
    """Get the device where model inputs should be placed.
    
    For models loaded with device_map="auto" (distributed across GPUs),
    inputs should go to the device of the first layer (embedding layer).
    """
    # For models with device_map="auto", get device of first parameter
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback_device)

def get_final_logit_prob(prompt, model, tokenizer, device = 'cuda', is_chat=False, has_system_role=False):
    with torch.no_grad():
        if is_chat and has_system_role:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True,return_tensors="pt", tokenize=True, return_dict=False)[0].tolist()
        elif is_chat:
            message = [
                {"role": "user", "content": prompt},]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True,return_tensors="pt", tokenize=True, return_dict=False)[0].tolist()
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        # Use device of first parameter for proper placement with device_map="auto"
        input_device = get_model_input_device(model, device)
        # Disable cache to avoid device mismatch issues with device_map="auto"
        outputs = model(torch.tensor([input_ids]).to(input_device), output_hidden_states=True, use_cache=False)
    #     response = model.generate(
    #         input_ids=torch.tensor([input_ids]).to(device),max_new_tokens=10,do_sample=False)
    #     decoded_response = tokenizer.decode(response[0][len(input_ids):], skip_special_tokens=False)
    #     print(f"decoded_response::{decoded_response}::")
    # print("outputs.logits.shape",outputs.logits.shape)
    model_log_probs = (
            outputs.logits[..., :]
            .log_softmax(-1)
            .squeeze()
            .detach()
            .cpu()
            .float()
        )
    # print("model_log_probs.shape",model_log_probs.shape)
    # print(model_log_probs.shape) # (seq_len, vocab_size)
    # print(torch.exp(model_log_probs).shape, type(torch.exp(model_log_probs))) # (seq_len, vocab_size)
    model_log_probs = model_log_probs[-1, :]
    # print(model_log_probs.shape) # (seq_len, vocab_size)
    # raise ValueError("!?")
    # get the maximum indixe of model_log_probs
    # max_ind = torch.argmax(model_log_probs)
    # print("max_ind:", max_ind, torch.exp(model_log_probs[max_ind]))
    # print(f"max token--{tokenizer.decode([max_ind])}--")
    # print(torch.sum(torch.exp(model_log_probs)))
    return torch.exp(model_log_probs)

def get_response(prompt, model, tokenizer, device = 'cuda', is_chat=False):

    with torch.no_grad():
        if is_chat:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True,return_tensors="pt", tokenize=True, return_dict=False)[0].tolist()
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        # outputs = model(torch.tensor([input_ids]).to(device), output_hidden_states=True)

        # Use device of first parameter for proper placement with device_map="auto"
        input_device = get_model_input_device(model, device)
        response = model.generate(
            input_ids=torch.tensor([input_ids]).to(input_device),
            attention_mask=torch.ones(1,len(input_ids)).to(input_device),
            max_new_tokens=10,do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
        decoded_response = tokenizer.decode(response[0][len(input_ids):], skip_special_tokens=True)
        # print(f"decoded_response::{decoded_response}::")
    return decoded_response

def get_completion_token_logprobs(prompt, completion, model, tokenizer, device='cuda', is_chat=False, has_system_role=False, include_eos=False):
    """Return per-token log probabilities for the given completion conditioned on the prompt.

    - If is_chat is False: tokenize `prompt` and `completion` separately and concatenate.
    - If is_chat is True: build the chat prompt prefix using the chat template with
      add_generation_prompt=True (system+user, assistant prefix), then append tokenized
      `completion` (no special tokens) and score each completion token autoregressively.
    - If include_eos is True: append the EOS token to the completion and include
      log P(EOS | prompt, completion) in the returned scores.
    """
    # Use device of first parameter for proper placement with device_map="auto"
    model_device = get_model_input_device(model, device)
    with torch.no_grad():
        if is_chat and has_system_role:
            # Build the chat prompt prefix (system + user + assistant prefix)
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            prefix_ids = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=False,
            )[0]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if include_eos and tokenizer.eos_token_id is not None:
                completion_ids = completion_ids + [tokenizer.eos_token_id]
            input_ids = torch.tensor([prefix_ids.tolist() + completion_ids])
            prefix_len = prefix_ids.shape[0]
        elif is_chat:
            # Build the chat prompt prefix (user + assistant prefix)
            message = [
                {"role": "user", "content": prompt},
            ]
            prefix_ids = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=False,
            )[0]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if include_eos and tokenizer.eos_token_id is not None:
                completion_ids = completion_ids + [tokenizer.eos_token_id]
            input_ids = torch.tensor([prefix_ids.tolist() + completion_ids])
            prefix_len = prefix_ids.shape[0]
        else:
            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if include_eos and tokenizer.eos_token_id is not None:
                completion_ids = completion_ids + [tokenizer.eos_token_id]
            # If prompt produces no tokens (e.g., Qwen with empty string — no BOS token),
            # prepend eos/pad token as minimal context so the first completion token has
            # something to condition on.
            if len(prompt_ids) == 0:
                fallback_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
                prompt_ids = torch.tensor([fallback_id])
            input_ids = torch.tensor([prompt_ids.tolist() + completion_ids])
            prefix_len = prompt_ids.shape[0]

        # Disable cache to avoid device mismatch issues with device_map="auto"
        outputs = model(input_ids.to(model_device), use_cache=False)
        log_probs = outputs.logits.log_softmax(-1).squeeze(0)  # [seq_len, vocab]

        # For k-th completion token, score is log_probs[prefix_len + k - 1, completion_ids[k]]
        token_logprobs = []
        for k, tok_id in enumerate(completion_ids):
            timestep_index = prefix_len + k - 1
            if timestep_index < 0 or timestep_index >= log_probs.shape[0]:
                # Should not happen, but guard anyway
                token_logprobs.append(torch.tensor(float("nan")))
            else:
                token_logprobs.append(log_probs[timestep_index, tok_id].detach().cpu().float())

        return torch.stack(token_logprobs)

def get_L_prompt(task, split_type, seed, sample_negative=True, variation=0, v2=True):
    # Check task registry first (for new extensible tasks)
    task_config = get_task(task)
    if task_config is not None:
        # NEW PATH: Use registered task configuration
        L_train, L_test = task_config['load_data'](
            seed=seed,
            split_type=split_type,
            sample_negative=sample_negative,
            v2=v2
        )
        # Wrap make_prompt to pass through v2 and variation
        base_make_prompt = task_config['make_prompt']
        def make_prompt(*args, **kwargs):
            return base_make_prompt(*args, v2=v2, variation=variation, **kwargs)
        return L_train, L_test, make_prompt

    # LEGACY PATH: Existing task implementations (unchanged)
    if task=='hypernym':
        L = load_noun_pair_data()
        if split_type=='hyper':
            L_train, L_test = split_train_test_no_overlap(L, seed=seed)
        elif split_type=='random':
            L_train, L_test = split_train_test(L, seed=seed, subsample=False, num_train=3000)
        elif split_type=='both':
            L_train, L_test = split_train_test_no_overlap_both(L, seed=2)
        else:
            raise ValueError("Wrong value for split-type")
        # Create a wrapper that includes the variation parameter
        def make_prompt(*args, **kwargs):
            return make_prompt_hypernymy(*args, variation=variation, **kwargs)
        make_prompt = make_prompt
    elif task.startswith('hypernym-'):
        # Dynamic hypernym task for any noun: hypernym-cars, hypernym-fruit, etc.
        noun = task.split('hypernym-', 1)[1]
        L_train, L_test = load_hypernym_noun_data(noun, v2=v2)
        # Use v2 prompt function if v2 mode, otherwise original
        if v2:
            def make_prompt(*args, **kwargs):
                return make_prompt_hypernymy_v2(*args, variation=variation, **kwargs)
        else:
            def make_prompt(*args, **kwargs):
                return make_prompt_hypernymy(*args, variation=variation, **kwargs)
        make_prompt = make_prompt
    elif task=='trivia-qa':
        L_train, L_test = load_triviaqa_data(seed=seed, sample_negative=sample_negative)
        make_prompt = make_prompt_triviaqa
    elif task=='swords':
        L_train, L_test = load_swords_data(seed=0)
        make_prompt = make_prompt_swords
    elif task=='lambada':
        L_train, L_test = load_lambada_data(seed=0, sample_negative=sample_negative)
        make_prompt = make_prompt_lambada
    elif task=='ifeval':
        L_train, L_test = load_ifeval_data(seed=0)
        make_prompt = make_prompt_ifeval
    elif task=='collie':
        L_train, L_test = load_collie_data(seed=0)
        make_prompt = make_prompt_collie
    else:
        raise NotImplementedError("Not a task")
    return L_train, L_test, make_prompt
