from pathlib import Path
import os
import sys
import argparse
from tqdm import tqdm
import torch
import gc
import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils import get_L_prompt, get_final_logit_prob, get_completion_token_logprobs
from logitlens import compute_logodds_final_layer, get_logodds_gen, get_logodds_disc

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()
yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
no_words = ["No", " No", "NO", "no", " no"]


def init_model(model_name, device):
    global model
    global tokenizer
    global terminators 
    torch_dtype = "auto"#torch.bfloat16
    # if 'gemma-3' in model_name:
    #     model = Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    # el
    if 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    print("model.config.torch_dtype:", model.config.torch_dtype)  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in model_name:
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


def get_base_model_name(modelname):
    modelname = modelname.split("output")[-1]
    return modelname.replace('/', '-')

def get_labels(task, LL):
    """Extract binary labels (1=positive, 0=negative) from data."""
    if task=='hypernym':
        return [1 if i.taxonomic.strip().capitalize() == 'Yes' else 0 for i in LL]
    elif task=="trivia-qa":
        if 'correct' in LL[0]:
            return [1 if i['correct'] == 'Yes' else 0 for i in LL]
        else:
            return [1 for i in LL]
    elif task=='swords':
        return [1 if i.synonym.capitalize() == 'Yes' else 0 for i in LL]
    elif task=='lambada':
        if 'correct' in LL[0]:
            return [1 if i['correct'] == 'Yes' else 0 for i in LL]
        else:
            return [1 for i in LL]
    elif task=='ifeval':
        return [1 if i['correct'] == 'Yes' else 0 for i in LL]
    elif task=='collie':
        return [1 if i['satisfies_constraint'] else 0 for i in LL]
    else:
        raise ValueError(f"Unknown task: {task}")

def create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logodds'):
    """
    Create scatter plot of generator vs validator log-odds or log-probs.
    
    Args:
        logodds_gen: List of generator log-odds/log-probs
        logodds_disc: List of discriminator/validator log-odds/log-probs
        labels: List of binary labels (1=positive, 0=negative)
        modelname: Model name for filename
        task: Task name for filename
        args: Command line arguments
        metric_type: 'logodds' or 'logprobs' for axis labels and filename
    """
    # Convert to numpy arrays
    logodds_gen_np = np.array([float(x) for x in logodds_gen])
    logodds_disc_np = np.array([float(x) for x in logodds_disc])
    labels_np = np.array(labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot positive examples (orange)
    pos_mask = labels_np == 1
    plt.scatter(logodds_gen_np[pos_mask], logodds_disc_np[pos_mask], 
                c='orange', label='Positive', alpha=0.6, s=30)
    
    # Plot negative examples (blue)
    neg_mask = labels_np == 0
    plt.scatter(logodds_gen_np[neg_mask], logodds_disc_np[neg_mask],
                c='blue', label='Negative', alpha=0.6, s=30)
    
    # Styling based on metric type
    metric_label = 'log-odds' if metric_type == 'logodds' else 'log-probs'
    plt.xlabel(f'Generator {metric_label}', fontsize=12)
    plt.ylabel(f'Validator {metric_label}', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Class', fontsize=10, title_fontsize=11)
    
    # Generate filename with metric type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = modelname.split('/')[-1].replace('--', '_')
    split = "train" if args.train else "test"
    filename = f"../outputs/viz_{model_short}_{task}_{split}_{metric_type}_{timestamp}.png"
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {filename}")
    plt.close()

def main(args):
    task = args.task
    modelname = args.model
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    print(f"gen_shots: {gen_shots}, disc_shots: {disc_shots}")

    train_flag = args.train
    split_type = args.split_type

    L_train, L_test, make_prompt = get_L_prompt(task, split_type, seed, sample_negative = args.sample_negative, variation = args.variation)
    print("Loaded data with negative_sample = {}!".format(args.sample_negative))
    device = get_device()
    print(f"Using device: {device}")

    init_model(modelname, device)
    


    #NOTE assume we just do llama or gemma. Same situation in both:
    first_sw_token = 2

    model_is_chat = False
    if 'instruct' in modelname.lower():
        model_is_chat = True
        first_sw_token = 1
        print("Model is chat model!")
    if "gpt" in modelname.lower():
        raise ValueError("If you are using GPT then rewrite this bit!")

    yestoks = [tokenizer.encode(i)[-1] for i in yes_words]
    notoks = [tokenizer.encode(i)[-1] for i in no_words]

    if train_flag:
        LL = L_train
        train_suffix = "--train"
    else:
        LL = L_test
        train_suffix = ""

    # Filter for single-token completions if requested
    if args.single_token_only:
        print(f"Original dataset size: {len(LL)}")
        filtered_LL = []
        for item in LL:
            gen_prompt_obj = make_prompt(item, style="generator", shots=gen_shots)
            completion_tokens = tokenizer.encode(gen_prompt_obj.completion, add_special_tokens=False)
            if len(completion_tokens) == 1:
                filtered_LL.append(item)
        LL = filtered_LL
        print(f"Filtered to single-token completions: {len(LL)}")
        train_suffix += "--single-token"

    if split_type=='random':
        split_suffix = ""
    elif split_type=='hyper':
        split_suffix = "--hyper"
    elif split_type=='both':
        split_suffix = "--both"
    else:
        raise ValueError()
    
    P_gen = []
    P_disc = []
    gen_sum_logprobs = []

    json_list = []
    # LL = LL[:10]
    for item in tqdm(LL):
        gen_obj = make_prompt(item, style='generator', shots=gen_shots)
        prompt_gen = gen_obj.prompt
        completion_gen = gen_obj.completion
        prompt_disc = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        probs_gen = get_final_logit_prob(prompt_gen, model, tokenizer, device, is_chat = model_is_chat) # TODO: change is_chat to True if instruction-tuned model
        P_gen.append(probs_gen)
        # Compute summed generator log-prob across all completion tokens (conditioned autoregressively)
        if args.use_full_completion_logprobs:
            gen_token_logprobs = get_completion_token_logprobs(prompt_gen, completion_gen, model, tokenizer, device, is_chat=model_is_chat)
            gen_sum_logprobs.append(float(gen_token_logprobs.sum().item()))
        probs_disc = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat = model_is_chat) # TODO: change is_chat to True if instruction-tuned model
        
        # # DEBUG: Print prompts and probabilities for first 5 examples
        # if len(P_disc) < 5:
        #     # Get Yes/No tokens
        #     if len(P_disc) == 0:
        #         yes_token_strings = ['Yes', ' Yes', 'yes', ' yes']
        #         no_token_strings = ['No', ' No', 'no', ' no']
        #         yestoks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_token_strings]
        #         notoks = [tokenizer.encode(s, add_special_tokens=False)[0] for s in no_token_strings]
        #         print("\n" + "="*80)
        #         print(f"DEBUG: use_full_completion_logprobs = {args.use_full_completion_logprobs}")
        #         print("="*80)
        #     
        #     p_yes = probs_disc[yestoks].sum()
        #     p_no = probs_disc[notoks].sum()
        #     log_odds = float(torch.log(p_yes / p_no))
        #     log_prob_yes = float(torch.log(p_yes))
        #     
        #     # Get ground truth
        #     if task == 'collie':
        #         gt_label = "POSITIVE" if item['satisfies_constraint'] else "NEGATIVE"
        #     elif task == 'hypernym':
        #         gt_label = "POSITIVE" if item.taxonomic.strip().capitalize() == 'Yes' else "NEGATIVE"
        #     else:
        #         gt_label = "UNKNOWN"
        #     
        #     print(f"\n--- Example {len(P_disc)} (Ground Truth: {gt_label}) ---")
        #     print(f"Discriminator Prompt:\n{prompt_disc}")
        #     print(f"\nProbabilities:")
        #     print(f"  P(Yes)={p_yes:.6f}, P(No)={p_no:.6f}, P(other)={1-p_yes-p_no:.6f}")
        #     print(f"  log-odds={log_odds:.4f}, log-prob={log_prob_yes:.4f}")
        
        P_disc.append(probs_disc)
        if args.train:
            prefix = " " if not model_is_chat else ""
            if task == 'hypernym':
                json_list.append({"noun1":item.noun1, "noun2":item.noun2, "taxonomic":item.taxonomic, "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0, 
                                "generator-completion": prefix + item.noun2.strip(), "discriminator-gold-completion": prefix + item.taxonomic.strip().capitalize()})
            elif task == 'trivia-qa':
                # print(item.keys())
                if args.sample_negative:
                    json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + item['correct']})
                else:
                    json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + 'Yes'})
            elif task == 'lambada':
                if args.sample_negative:
                    json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + item['correct']})
                else:
                    json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + 'Yes'})
            elif task == 'swords':
                json_list.append({"context":item.context, "target":item.target, "replacement":item.replacement, "synonym":item.synonym,
                                "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": item.replacement.strip(), "discriminator-gold-completion": prefix + item.synonym.strip().capitalize()})
            elif task == "ifeval":
                json_list.append({"prompt":item.prompt, "generator-completion":item.response, "discriminator-gold-completion":item.correct})
            else:
                raise NotImplementedError("Not a task")
            # print(json_list[-1])
    
    # Compute logodds for visualization/analysis (needed for both train and test)
    if args.use_full_completion_logprobs:
        # Multi-token case: use log-probs for both generator and discriminator
        logodds_gen = [torch.tensor(v) for v in gen_sum_logprobs]
        logodds_disc = [torch.log(torch.sum(P_disc[ii][..., yestoks], dim=-1)) for ii in range(len(P_disc))]
        logprobs_gen = None  # Not needed for multi-token
        logprobs_disc = None
    else:
        # Single-token case: compute both log-odds and log-probs for visualization
        logodds_gen = [get_logodds_gen(P_gen, LL, ii, tokenizer, first_sw_token, task, is_chat = model_is_chat, use_lgo=True) for ii in range(len(P_gen))]
        logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]
        # Also compute log-probs version for second plot
        logprobs_gen = [get_logodds_gen(P_gen, LL, ii, tokenizer, first_sw_token, task, is_chat = model_is_chat, use_lgo=False) for ii in range(len(P_gen))]
        logprobs_disc = [torch.log(torch.sum(P_disc[ii][..., yestoks], dim=-1)) for ii in range(len(P_disc))]
    
    # Compute confusion matrix for discriminator
    # Get ground truth labels (1=positive, 0=negative)
    true_labels = get_labels(task, LL)
    
    # Determine threshold based on metric type
    # For log-odds: threshold = 0 (since log(P(yes)/P(no)) = 0 when P(yes) = P(no) = 0.5)
    # For log-probs: threshold = log(0.5) ≈ -0.693 (since log(P(yes)) = log(0.5) when P(yes) = 0.5)
    if args.use_full_completion_logprobs:
        threshold = np.log(0.5)  # log-probs threshold
        metric_name = "log-probs"
    else:
        threshold = 0.0  # log-odds threshold
        metric_name = "log-odds"
    
    # Make predictions: predicted = 1 if score > threshold, else 0
    disc_scores = np.array([float(x) for x in logodds_disc])
    pred_labels = (disc_scores > threshold).astype(int)
    true_labels_np = np.array(true_labels)
    
    # Compute confusion matrix components
    tp = np.sum((pred_labels == 1) & (true_labels_np == 1))  # True Positives
    fp = np.sum((pred_labels == 1) & (true_labels_np == 0))  # False Positives
    tn = np.sum((pred_labels == 0) & (true_labels_np == 0))  # True Negatives
    fn = np.sum((pred_labels == 0) & (true_labels_np == 1))  # False Negatives
    
    # Print confusion matrix
    print(f"\nDiscriminator Confusion Matrix ({metric_name}, threshold={threshold:.3f}):")
    print(f"                 Predicted Positive  Predicted Negative")
    print(f"Actual Positive        {tp:6d}              {fn:6d}")
    print(f"Actual Negative        {fp:6d}              {tn:6d}")
    print(f"\nAccuracy: {(tp + tn) / len(true_labels):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    print(f"F1 Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}\n")
    
    # Debug: Save ALL discriminator and generator values with ground truth to CSV
    if args.debug_save_values:
        import csv
        debug_suffix = "logprobs" if args.use_full_completion_logprobs else "logodds"
        debug_file = f"../outputs/debug_values_{task}_{debug_suffix}.csv"
        
        with open(debug_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header - include task-specific fields for verification
            if task == 'hypernym':
                writer.writerow(['index', 'noun1', 'noun2', 'taxonomic', 'ground_truth', 'disc_score', 'gen_score'])
            elif task == 'swords':
                writer.writerow(['index', 'context', 'target', 'replacement', 'synonym', 'ground_truth', 'disc_score', 'gen_score'])
            elif task in ['trivia-qa', 'lambada']:
                writer.writerow(['index', 'ground_truth', 'disc_score', 'gen_score'])
            else:
                writer.writerow(['index', 'ground_truth', 'disc_score', 'gen_score'])
            
            # Write all data
            for i in range(len(logodds_disc)):
                if task == 'hypernym':
                    writer.writerow([
                        i,
                        LL[i].noun1,
                        LL[i].noun2,
                        LL[i].taxonomic,
                        true_labels[i],
                        float(logodds_disc[i]),
                        float(logodds_gen[i]) if i < len(logodds_gen) else ''
                    ])
                elif task == 'swords':
                    writer.writerow([
                        i,
                        LL[i].context,
                        LL[i].target,
                        LL[i].replacement,
                        LL[i].synonym,
                        true_labels[i],
                        float(logodds_disc[i]),
                        float(logodds_gen[i]) if i < len(logodds_gen) else ''
                    ])
                else:
                    writer.writerow([
                        i,
                        true_labels[i],
                        float(logodds_disc[i]),
                        float(logodds_gen[i]) if i < len(logodds_gen) else ''
                    ])
        print(f"Debug values saved to: {debug_file} ({len(logodds_disc)} examples)")
    
    if args.train:
        for jj in range(len(json_list)):
            json_list[jj]["generator-log-prob"] = float(logodds_gen[jj])
            json_list[jj]["discriminator-log-prob"] = float(logodds_disc[jj])
            # print(json_list[jj])
        print(f"Saving train data to ../data/{task}-train-{modelname.split('/')[-1]}.json")
        with open(f"../data/{task}-train-{modelname.split('/')[-1]}.json", 'w') as f:
            json.dump(json_list, f, indent=4)
        
        # Create visualization if requested
        if args.viz:
            labels = get_labels(task, LL)
            if args.use_full_completion_logprobs:
                # Multi-token: one plot with log-probs
                create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logprobs')
            else:
                # Single-token: two plots (log-odds and log-probs)
                create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logodds')
                create_visualization(logprobs_gen, logprobs_disc, labels, modelname, task, args, metric_type='logprobs')
        return

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    
    
    res_dict = compute_logodds_final_layer(task,
        P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks, is_chat=model_is_chat, gen_logprobs=(gen_sum_logprobs if args.use_full_completion_logprobs else None))

    
    basename = get_base_model_name(modelname)


    summary_file = os.path.join("../outputs/eval_results.csv")
    with open(summary_file, 'a') as f:
        # if file is empty:
        if os.stat(summary_file).st_size == 0:
            f.write("model,task,corr_all,corr_pos,corr_neg,disc_acc,disc_roc, gen_acc_5, gen_acc_10, gen_acc_40, gen_acc_100, gen_acc_1000,gen_mrr_pos, gen_mrr_neg, gen_shots,disc_shots,split,split_type,seed,spear_all,spear_pos,spear_neg,\n")
        split = "train" if args.train else "test"
        f.write(f"{modelname},{task},{res_dict['corr_all']},{res_dict['corr_pos']},{res_dict['corr_neg']},{res_dict['disc_acc']},{res_dict['disc_roc']},{res_dict['gen_acc_dict'][5]},{res_dict['gen_acc_dict'][10]},{res_dict['gen_acc_dict'][40]},{res_dict['gen_acc_dict'][100]},{res_dict['gen_acc_dict'][1000]},{res_dict['gen_mrr_pos']},{res_dict['gen_mrr_neg']},{gen_shots},{disc_shots},{split},{split_type},{seed}")
        f.write(f",{res_dict['spear_all']},{res_dict['spear_pos']},{res_dict['spear_neg']},\n")
    
    # Create visualization if requested
    if args.viz:
        labels = get_labels(task, LL)
        if args.use_full_completion_logprobs:
            # Multi-token: one plot with log-probs
            create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logprobs')
        else:
            # Single-token: two plots (log-odds and log-probs)
            create_visualization(logodds_gen, logodds_disc, labels, modelname, task, args, metric_type='logodds')
            create_visualization(logprobs_gen, logprobs_disc, labels, modelname, task, args, metric_type='logprobs')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log-odds on test data")
    parser.add_argument("--model", type=str, help="model directory to process (this should contain merged/ subdirectory) or hf model")
    # parser.add_argument("--tunedlens", action="store_true", default=False, help="")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling the data.")
    parser.add_argument("--disc-shots", type=str, default='few', help="'zero' vs 'few'")
    parser.add_argument("--gen-shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--train", action="store_true", default=False, help="log-odds of train or test set?")
    parser.add_argument("--split_type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument("--task", type=str, default='hypernym', help="hypernym, trivia-qa, etc")
    parser.add_argument("--sample_negative", action="store_true", default=False, help="whether to sample negative examples when loading trivia-qa or lambada")
    parser.add_argument("--variation", type=str, default="0", help="variation parameter for hypernym prompt formatting (default: '0')")
    parser.add_argument("--single_token_only", action="store_true", default=False, help="only use test data where generator completion is exactly one token")
    parser.add_argument("--use_full_completion_logprobs", action="store_true", default=False, help="use autoregressive log-probs over all completion tokens for generator scoring")
    parser.add_argument("--viz", action="store_true", default=False, help="create and save visualization plot of generator vs validator log-odds")
    parser.add_argument("--debug_save_values", action="store_true", default=False, help="save discriminator log-odds/log-probs values to file for debugging")

    args = parser.parse_args()
    main(args)
