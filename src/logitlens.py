"""
Implementation of logit lens using nnsight.
For more information, see https://nnsight.net/notebooks/tutorials/logit_lens/

"""

from nnsight import LanguageModel
import torch
import seaborn as sns
import numpy as np
import sklearn.metrics
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score

device = "cuda:0"


def load_model_nnsight(modelname, device):
    """Load the model into an nnsight.LanguageModel object

    Usage:
    model = load_model_nnsight("google/gemma-2b", "cuda:0")
    """
    model = LanguageModel(modelname, device_map=device, dispatch=True)
    return model


def get_logitlens_output(prompt, model, modelname_short):
    """
    Usage:
    model = load_model_nnsight("google/gemma-2b", "cuda:0")
    prompt = "The Eiffel Tower is in the city of"
    probs, max_probs, tokens, words, input_words = logitlens(prompt, model)
    """
    if modelname_short in ["gpt2-xl"]:
        layers = model.transformer.h
    if modelname_short in ["gemma-2-2b", "Meta-Llama-3-8B-Instruct"] or 'llama' in modelname_short.lower():
        layers = model.model.layers

    probs_layers = []

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                if modelname_short in ["gpt2-xl"]:
                    layer_output = model.lm_head(
                        model.transformer.ln_f(layer.output[0])
                    )
                elif modelname_short in [
                    "gemma-2-2b",
                    "Meta-Llama-3-8B-Instruct",
                    "Llama-3.2-3B-Instruct"
                ]:
                    layer_output = model.lm_head(model.model.norm(layer.output[0]))
                else:
                    raise NotImplementedError("Model not implemented.")

                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    probs = torch.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)
    #print(f"logitlens_output--max_ind:{tokens[-1, -1]}, {model.tokenizer.decode([tokens[-1, -1]])}, {max_probs[-1,-1]}")
    # Decode token IDs to words for each layer
    words = [
        [
            model.tokenizer.decode(t).encode("unicode_escape").decode()
            for t in layer_tokens
        ]
        for layer_tokens in tokens
    ]

    # Access the 'input_ids' attribute of the invoker object to get the input words
    # input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0]["input_ids"][0]]
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
    # print(len(input_words))
    # print(len(words), len(words[0]))
    # print(f"logitlens_output: {probs.shape}, {max_probs.shape}, {tokens.shape}, {words}, {input_words}")
    # torch.Size([26, 10, 256000]), torch.Size([26, 10]), torch.Size([26, 10])
    return probs, max_probs, tokens, words, input_words


def get_rank(L, ind):
    value = L[ind]
    sorted_L, sorted_indices = torch.sort(L, descending=True)
    rank = (sorted_L == value).nonzero(as_tuple=True)[0][0].item() + 1
    return rank


def get_rank_in_subset(L, ind, candidate_indices):
    """
    Get rank of token 'ind' among only the candidate tokens.
    
    Args:
        L: Probability/score distribution over full vocabulary
        ind: Target token index
        candidate_indices: List of token indices to consider (e.g., all dataset tokens)
    
    Returns:
        Rank of ind among candidate_indices (1-indexed)
    """
    # Extract scores for candidate tokens only
    candidate_scores = L[candidate_indices]
    
    # Sort in descending order
    sorted_scores, sorted_indices = torch.sort(candidate_scores, descending=True)
    
    # Find which candidate index corresponds to our target
    target_candidate_idx = (candidate_indices == ind).nonzero(as_tuple=True)[0]
    
    if len(target_candidate_idx) == 0:
        # Target token not in candidate set - return large rank
        return len(candidate_indices) + 1
    
    target_candidate_idx = target_candidate_idx[0].item()
    target_score = candidate_scores[target_candidate_idx]
    
    # Find rank among candidates
    rank = (sorted_scores >= target_score).sum().item()
    
    return rank


def get_logodds_disc(Ps, ii, yestoks, notoks):
    if notoks is None:
        lgo = torch.log(torch.sum(Ps[ii][..., yestoks], dim=-1)) 
    else:
        lgo = torch.log(torch.sum(Ps[ii][..., yestoks], dim=-1)) - torch.log(
            torch.sum(Ps[ii][..., notoks], dim=-1)
        )
    lgo[torch.isinf(lgo)] = 35  # truncate infs
    return lgo


def get_logodds_gen(Ps, L, ii, tokenizer, first_sw_token, task, is_chat = False, use_lgo=True):
    #TODO should clean up this code so it takes in the completion
    if is_chat:
        prefix = ""
    else:
        prefix = "a "
    if task=='hypernym':
        ind = tokenizer.encode(prefix + L[ii].noun2)[first_sw_token]
    elif task=='trivia-qa':
        ind = tokenizer.encode(prefix + L[ii]['answers'][0].capitalize())[first_sw_token]
    elif task=='swords':
        ind = tokenizer.encode(prefix + L[ii].replacement)[first_sw_token]
    elif task=='lambada':
        ind = tokenizer.encode(prefix + L[ii]['final_word'])[first_sw_token]
    elif task=='ifeval':
        ind = tokenizer.encode(prefix + L[ii]['response'])[first_sw_token:]
    elif task=='collie':
        ind = tokenizer.encode(prefix + L[ii]['generated'])[first_sw_token:]
    else:
        raise ValueError("!")
    
    logodds = []
    for i in ind if isinstance(ind, list) else [ind]:
        if use_lgo:
            lgo = torch.log(torch.abs(Ps[ii][..., i])) - torch.log(
                torch.abs(1 - Ps[ii][..., i])
            )
        else:
            lgo = torch.log(torch.abs(Ps[ii][..., i])) 
        lgo[torch.isinf(lgo)] = 35  # truncate infs
        logodds.append(lgo)
    
    return torch.stack(logodds).sum(dim=0)


def makepreds_disc(logodds, threshold=0, layer_disc=-1):
    if logodds[0].dim() == 0:
        pred = ["Yes" if x > threshold else "No" for x in [i.tolist() for i in logodds][:]]
    else:
        pred = ["Yes" if x > threshold else "No" for x in [i[layer_disc].tolist() for i in logodds][:]]
    return pred


def makepreds_gen(ranks, threshold=40):
    return ["Yes" if r <= threshold else "No" for r in ranks]


def compute_disc_accuracy(gold, logodds_disc):
    preds = [1 if i>0 else 0 for i in logodds_disc]
    disc_accuracy = sklearn.metrics.accuracy_score(gold, preds)
    # print("\naccuracy of discriminator fs: {}".format(disc_accuracy))

    # fpr, tpr, thresholds = roc_curve(gold, preds)
    if len(set(gold)) == 1:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(gold, logodds_disc)
    # print(f'roc_auc: {roc_auc}')
    return disc_accuracy, roc_auc

def compute_gen_roc(gold, logodds_gen):
    """Compute AUC-ROC using generator scores."""
    if len(set(gold)) == 1:
        # Only one class present, ROC is undefined
        return np.nan
    else:
        return roc_auc_score(gold, logodds_gen)

def compute_gen_accuracy(golds, ranks, thresholds = [5, 10, 40, 100, 1000], prefix=""):
    gen_accuracies = {} #map threshold to accuracy
    for t in thresholds:
        preds = [1 if r <= t else 0 for r in ranks]
        a = sklearn.metrics.accuracy_score(golds, preds)
        gen_accuracies[t]= a
        print("accuracy of generator {}: (th={}: {})".format(prefix, t, a))
    return gen_accuracies

def compute_gen_mrr(golds, ranks):
    ranks_pos = [r for i, r in enumerate(ranks) if golds[i] == 1]
    ranks_neg = [r for i, r in enumerate(ranks) if golds[i] == 0]
    mrr_pos = np.mean([1 / r for r in ranks_pos])
    mrr_neg = np.mean([1 / r for r in ranks_neg])
    return mrr_pos, mrr_neg

def compute_metrics(task, L, logodds_gen, logodds_disc, ranks, ranks_dataset=None):
    if task=='hypernym':
        golds = [1 if i.taxonomic.strip().capitalize() == 'Yes' else 0 for i in L]
    elif task=="trivia-qa":
        if 'correct' in L[0]:
            golds = [1 if i['correct'] == 'Yes' else 0 for i in L]
        else:
            golds = [1  for i in L]
    elif task=='swords':
        golds = [1 if i.synonym.capitalize() == 'Yes' else 0 for i in L]
    elif task=='lambada':
        if 'correct' in L[0]:
            golds = [1 if i['correct'] == 'Yes' else 0 for i in L]
        else:
            golds = [1  for i in L]
    elif task=='ifeval':
        golds = [1 if i['correct'] == 'Yes' else 0 for i in L]
    elif task=='collie':
        golds = [1 if i['satisfies_constraint'] else 0 for i in L]
    else:
        raise ValueError("!")

    print("correlation: zs gen, fs disc (more usual)")
    corr_all = pearsonr(logodds_gen, logodds_disc).statistic
    spear_all = spearmanr(logodds_gen, logodds_disc).statistic
    logodds_gen_pos = [logodds_gen[i] for i in range(len(logodds_gen)) if golds[i] == 1]
    logodds_gen_neg = [logodds_gen[i] for i in range(len(logodds_gen)) if golds[i] == 0]
    logodds_disc_pos = [logodds_disc[i] for i in range(len(logodds_disc)) if golds[i] == 1]
    logodds_disc_neg = [logodds_disc[i] for i in range(len(logodds_disc)) if golds[i] == 0]
    corr_pos = pearsonr(logodds_gen_pos, logodds_disc_pos).statistic
    spear_pos = spearmanr(logodds_gen_pos, logodds_disc_pos).statistic
    if len(logodds_gen_neg) == 0:
        corr_neg = np.nan
        spear_neg = np.nan
    else:
        corr_neg = pearsonr(logodds_gen_neg, logodds_disc_neg).statistic
        spear_neg = spearmanr(logodds_gen_neg, logodds_disc_neg).statistic
    print(f"correlation: all = {corr_all}, pos = {corr_pos}, neg = {corr_neg}")
    print(f"spearman: all = {spear_all}, pos = {spear_pos}, neg = {spear_neg}")
    disc_acc, disc_roc = compute_disc_accuracy(golds, logodds_disc)
    print(f"disc_acc: {disc_acc}, disc_roc: {disc_roc}")
    
    # Compute generator ROC
    gen_roc = compute_gen_roc(golds, logodds_gen)
    print(f"gen_roc: {gen_roc}")

    print("\n--- Full Vocabulary Ranking ---")
    gen_acc_dict = compute_gen_accuracy(golds, ranks, thresholds = [5, 10, 40, 100, 1000], prefix="full-vocab")
    gen_mrr_pos, gen_mrr_neg = compute_gen_mrr(golds, ranks)
    print(f"gen_mrr_pos (full-vocab): {gen_mrr_pos}, gen_mrr_neg (full-vocab): {gen_mrr_neg}")

    # Compute dataset-constrained metrics if provided
    result = {
        'corr_all': corr_all,
        'corr_pos': corr_pos,
        'corr_neg': corr_neg,
        'disc_acc': disc_acc,
        'disc_roc': disc_roc,
        'gen_roc': gen_roc,
        'gen_acc_dict': gen_acc_dict,
        'gen_mrr_pos': gen_mrr_pos,
        'gen_mrr_neg': gen_mrr_neg,
        'spear_all': spear_all,
        'spear_pos': spear_pos,
        'spear_neg': spear_neg
    }
    
    if ranks_dataset is not None:
        print("\n--- Dataset-Constrained Ranking ---")
        gen_acc_dict_dataset = compute_gen_accuracy(golds, ranks_dataset, thresholds = [5, 10, 40, 100, 1000], prefix="dataset")
        gen_mrr_pos_dataset, gen_mrr_neg_dataset = compute_gen_mrr(golds, ranks_dataset)
        print(f"gen_mrr_pos (dataset): {gen_mrr_pos_dataset}, gen_mrr_neg (dataset): {gen_mrr_neg_dataset}")
        
        result['gen_acc_dict_dataset'] = gen_acc_dict_dataset
        result['gen_mrr_pos_dataset'] = gen_mrr_pos_dataset
        result['gen_mrr_neg_dataset'] = gen_mrr_neg_dataset
    
    return result




def compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=-1, layer_disc=-1):

    if task=='hypernym':
        gold = [i.taxonomic.capitalize() for i in L]
    elif task=="trivia-qa":
        if 'correct' in L[0]:
            gold = [i['correct'] for i in L]
        else:
            gold = ['Yes' for i in L]
    elif task=='swords':
        gold = [i.synonym.capitalize() for i in L]
    elif task=='lambada':
        if 'correct' in L[0]:
            gold = [i['correct'] for i in L]
        else:
            gold = ['Yes' for i in L]
    else:
        raise ValueError("!")

    print("correlation: zs gen, fs disc (more usual)")
    # print(logodds_gen[0].size)
    # print(type(logodds_gen[0]) == torch.Tensor)
    if logodds_gen[0].dim() == 0:
        corr = pearsonr(
            [i.tolist() for i in logodds_gen],
            [i.tolist() for i in logodds_disc]
        ).statistic
    else:
        corr = pearsonr(
            [i[layer_gen].tolist() for i in logodds_gen],
            [i[layer_disc].tolist() for i in logodds_disc]
        ).statistic
    print(corr)

    disc_accuracy = sklearn.metrics.accuracy_score(gold, makepreds_disc(logodds_disc, threshold=0, layer_disc=layer_disc))
    print("\naccuracy of discriminator fs: {}".format(disc_accuracy))
    # disc_accuracy_new, roc = compute_disc_accuracy(task, L, logodds_disc)
    # print("\naccuracy of discriminator fs (new): {}, roc{}".format(disc_accuracy_new, roc))

    gen_accuracies = {} #map threshold to accuracy
    for threshold in [5, 10, 40, 100, 1000]:
        a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=threshold))
        gen_accuracies[threshold]= a
        print("accuracy of generator zs: (th={}: {})".format(threshold, a))
    return disc_accuracy, gen_accuracies, corr


# def compute_logodds(
#     task, P_gen, P_disc, L, tokenizer, first_sw_token, yestoks, notoks, layer_gen=-1, layer_disc=-1
# ):

#     if task=='hypernym':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii].noun2)[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='trivia-qa':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii]['answers'][0])[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='swords':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii].replacement)[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='lambada':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii]['final_word'])[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     else:
#         raise ValueError("!!")

#     logodds_gen = [get_logodds_gen(P_gen, L, ii, tokenizer, first_sw_token, task) for ii in range(len(P_gen))]
#     logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]

#     disc_accuracy, gen_accuracies, corr = compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=layer_gen, layer_disc=layer_disc)
#     res_dict = compute_metrics(task, L, logodds_gen, logodds_disc, ranks)
#     # return res_dict
#     return ranks, logodds_gen, logodds_disc, corr


def extract_dataset_tokens(task, L, tokenizer, first_sw_token, is_chat=False):
    """Extract all unique completion token IDs from the dataset."""
    prefix = "a " if not is_chat else ""
    unique_tokens = set()
    
    if task=='hypernym':
        for item in L:
            token_ids = tokenizer.encode(prefix + item.noun2, add_special_tokens=False)
            if len(token_ids) > first_sw_token:
                unique_tokens.add(token_ids[first_sw_token])
    elif task=='trivia-qa':
        for item in L:
            # Add both lowercase and capitalized versions
            token_ids = tokenizer.encode(prefix + item['answers'][0], add_special_tokens=False)
            if len(token_ids) > first_sw_token:
                unique_tokens.add(token_ids[first_sw_token])
            token_ids_cap = tokenizer.encode(prefix + item['answers'][0].capitalize(), add_special_tokens=False)
            if len(token_ids_cap) > first_sw_token:
                unique_tokens.add(token_ids_cap[first_sw_token])
    elif task=='swords':
        idx = first_sw_token if is_chat else first_sw_token - 1
        for item in L:
            token_ids = tokenizer.encode(item.replacement if is_chat else prefix + item.replacement, add_special_tokens=False)
            if len(token_ids) > idx:
                unique_tokens.add(token_ids[idx])
    elif task=='lambada':
        for item in L:
            token_ids = tokenizer.encode(prefix + item['final_word'], add_special_tokens=False)
            if len(token_ids) > first_sw_token:
                unique_tokens.add(token_ids[first_sw_token])
    elif task in ['ifeval', 'collie']:
        # For multi-token tasks, collect all tokens
        prefix = ""
        key = 'response' if task == 'ifeval' else 'generated'
        for item in L:
            token_ids = tokenizer.encode(prefix + item[key], add_special_tokens=False)
            for tid in token_ids[first_sw_token:]:
                unique_tokens.add(tid)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Convert to sorted tensor for consistent indexing
    import torch
    return torch.tensor(sorted(unique_tokens), dtype=torch.long)


def compute_logodds_final_layer(
    task, P_gen, P_disc, L, tokenizer, first_sw_token, yestoks, notoks, is_chat = False, gen_logprobs=None, corrected_logodds_gen=None):

    prefix = "a " if not is_chat else ""
    
    # Extract all unique completion tokens from dataset
    print("Extracting dataset tokens...")
    dataset_token_ids = extract_dataset_tokens(task, L, tokenizer, first_sw_token, is_chat)
    print(f"  Found {len(dataset_token_ids)} unique tokens in dataset completions")
    
    # Compute full-vocabulary ranks
    print("Computing full-vocabulary ranks...")
    if task=='hypernym':
        # for ii in range(len(P_gen)):
        #     print(f'--compute-logodds, i=0:P:{P_gen[ii].shape}')
        #     print(f'--compute-logodds, i=0:P:{P_gen[ii].shape}')
        #     print(f'--compute-logodds, i=0:L:{L[ii]}')
        #     print(f'--compute-logodds, i=0:L:{L[ii].noun2}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.encode("a " + L[ii].noun2)}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.tokenize("a " + L[ii].noun2)}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.encode("a " + L[ii].noun2)[first_sw_token]}')

        ranks = [
            get_rank(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii].noun2)[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task=='trivia-qa':
        ranks = [
            min(
                get_rank(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0])[first_sw_token]
                ),
                get_rank(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0].capitalize())[first_sw_token]
                )
            )
            
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task=='swords':
        if is_chat:
            ranks = [
                get_rank(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token]
                )
                for ii in tqdm(range(len(P_gen)))
            ]
        else:
            ranks = [
                get_rank(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token-1]
                )
                for ii in tqdm(range(len(P_gen)))
            ]
    elif task == 'lambada':
        ranks = [
            get_rank(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii]['final_word'])[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task == 'ifeval':
        prefix = ""
        tokens = [tokenizer.encode(prefix + L[ii]['response'])[first_sw_token:] for ii in range(len(P_gen))]
        ranks = [
            sum(get_rank(P_gen[ii][:], t) for t in tokens[ii]) / len(tokens[ii])
            if len(tokens[ii]) > 0 else float('inf')
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task == 'collie':
        prefix = ""
        tokens = [tokenizer.encode(prefix + L[ii]['generated'])[first_sw_token:] for ii in range(len(P_gen))]
        ranks = [
            sum(get_rank(P_gen[ii][:], t) for t in tokens[ii]) / len(tokens[ii])
            if len(tokens[ii]) > 0 else float('inf')
            for ii in tqdm(range(len(P_gen)))
        ]
    else:
        raise ValueError("!!")

    # Compute dataset-constrained ranks (only among dataset completion tokens)
    print("Computing dataset-constrained ranks...")
    if task=='hypernym':
        ranks_dataset = [
            get_rank_in_subset(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii].noun2)[first_sw_token], dataset_token_ids
            )
            for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
        ]
    elif task=='trivia-qa':
        ranks_dataset = [
            min(
                get_rank_in_subset(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0])[first_sw_token], dataset_token_ids
                ),
                get_rank_in_subset(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0].capitalize())[first_sw_token], dataset_token_ids
                )
            )
            for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
        ]
    elif task=='swords':
        idx = first_sw_token if is_chat else first_sw_token - 1
        if is_chat:
            ranks_dataset = [
                get_rank_in_subset(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token], dataset_token_ids
                )
                for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
            ]
        else:
            ranks_dataset = [
                get_rank_in_subset(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token-1], dataset_token_ids
                )
                for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
            ]
    elif task == 'lambada':
        ranks_dataset = [
            get_rank_in_subset(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii]['final_word'])[first_sw_token], dataset_token_ids
            )
            for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
        ]
    elif task == 'ifeval':
        prefix = ""
        tokens = [tokenizer.encode(prefix + L[ii]['response'])[first_sw_token:] for ii in range(len(P_gen))]
        ranks_dataset = [
            sum(get_rank_in_subset(P_gen[ii][:], t, dataset_token_ids) for t in tokens[ii]) / len(tokens[ii])
            if len(tokens[ii]) > 0 else float('inf')
            for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
        ]
    elif task == 'collie':
        prefix = ""
        tokens = [tokenizer.encode(prefix + L[ii]['generated'])[first_sw_token:] for ii in range(len(P_gen))]
        ranks_dataset = [
            sum(get_rank_in_subset(P_gen[ii][:], t, dataset_token_ids) for t in tokens[ii]) / len(tokens[ii])
            if len(tokens[ii]) > 0 else float('inf')
            for ii in tqdm(range(len(P_gen)), desc="Dataset ranks")
        ]
    else:
        raise ValueError("!!")

    # Use corrected_logodds_gen if provided (for typicality correction)
    if corrected_logodds_gen is not None:
        logodds_gen = [float(v) for v in corrected_logodds_gen]
        logodds_disc = [torch.log(torch.sum(P_disc[ii][..., yestoks], dim=-1)) for ii in range(len(P_disc))]
    elif gen_logprobs is not None:
        # Multi-token case: use log-probs for both generator and discriminator
        logodds_gen = [float(v) for v in gen_logprobs]
        logodds_disc = [torch.log(torch.sum(P_disc[ii][..., yestoks], dim=-1)) for ii in range(len(P_disc))]
    else:
        # Single-token case: use log-odds for both generator and discriminator
        logodds_gen = [get_logodds_gen(P_gen, L, ii, tokenizer, first_sw_token, task, is_chat=is_chat, use_lgo=True) for ii in range(len(P_gen))]
        logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]

    # disc_accuracy, gen_accuracies, corr = compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks)
    res_dict = compute_metrics(task, L, logodds_gen, logodds_disc, ranks, ranks_dataset=ranks_dataset)
    print(res_dict)
    return res_dict

