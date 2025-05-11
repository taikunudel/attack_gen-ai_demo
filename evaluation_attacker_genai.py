#!/usr/bin/env python
# coding: utf-8
from typing import List, Dict, Any, Tuple
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
import os
import sys
import GPUtil
import hashlib
import diskcache
cache = diskcache.Cache('/usa/taikun/07_transencoder/attack-genai', size_limit=10e9)
cache.stats(enable=True)    
from collections import defaultdict # Added this import
from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu
from itertools import product
import random
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BertForMaskedLM, 
    BertConfig)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset, 
    Dataset, 
    DataLoader, 
    Subset, 
    RandomSampler)
import tensorflow as tf
# print("TensorFlow sees the following Physical GPUs:", tf.config.list_physical_devices('GPU'))
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:                          
#     tf.config.experimental.set_memory_growth(gpus[1], True)   # ❶
import tensorflow_hub as hub
from train_attacker_genai import *
# import get_raw_logits

def getUSEcosSimilarity(srcDocs, copyDocs, embed):
    USEcosinSimilarity = []
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    for src, copy in zip(srcDocs, copyDocs):
        emb1, emb2 = embed([src, copy])["outputs"]
        emb1, emb2 = torch.tensor(emb1.numpy()), torch.tensor(emb2.numpy())
        srcEmb = torch.unsqueeze(emb1, dim=0) # [embSz] -> [1, embSz]
        advEmb = torch.unsqueeze(emb2, dim=0)
        es = sim_metric(srcEmb, advEmb)
        USEcosinSimilarity.append(es.item())
    return USEcosinSimilarity

def get_influences(true_class_id, predictions, probs):
    '''
    Calculate influence scores for binary classification
    '''
    # Fix typo in variable name
    influences = []
    
    # Get source probability for true class
    src_prob = probs[0] if predictions[0] == true_class_id else 1 - probs[0]
    
    # Calculate influence for each prediction
    for i, pred in enumerate(predictions):
        curr_prob = probs[i] if pred == true_class_id else 1 - probs[i]
        influence = curr_prob - src_prob
        influences.append(influence)  # Fixed typo in append
        
    return influences

def process_single_document(input_id, attention_mask, model, true_class_id, tokenizer, num_doc_masks, mask_token_id, server_url):
    # Convert tensors to lists for hashing the input
    input_id_list = input_id.cpu().tolist()
    attention_mask_list = attention_mask.cpu().tolist()
    true_class_id_item = true_class_id.item()

    # Create the cache key based on the input parameters
    input_key = hashlib.sha256(json.dumps({
        "input_id_list": input_id_list,
        "attention_mask_list": attention_mask_list,
        "true_class_id_item": true_class_id_item,
        "num_doc_masks": num_doc_masks,
        "mask_token_id": mask_token_id,
    }, sort_keys=True).encode()).hexdigest()

    # Check if the result for this input is already in the cache
    if input_key in cache:
        cached_result = cache[input_key]
        # Ensure the retrieved tensor is on the correct device
        masked_input = torch.from_numpy(cached_result[0]).to(input_id.device)
        positions = cached_result[1]
        return masked_input, positions

    # Find valid positions (non-padding tokens)
    valid_positions = [i for i, val in enumerate(attention_mask_list) if val == 1]

    # Skip if not enough valid positions
    if len(valid_positions) < num_doc_masks:
        num_doc_masks = len(valid_positions) - 1

    # Create batch inputs for this sample
    sample_size = len(valid_positions) + 1  # +1 for original
    sample_input_ids = [input_id_list.copy() for _ in range(sample_size)]
    for j, pos in enumerate(valid_positions):
        sample_input_ids[j + 1][pos] = mask_token_id

    sample_input_tensor = torch.tensor(sample_input_ids)
    sample_docs = tokenizer.batch_decode(sample_input_tensor, skip_special_tokens=True)

    # Run batch inference
    _, predictions_, probs_ = get_raw_logits.process_file(data=sample_docs, server_url=server_url)

    # Calculate token influences
    influences = get_influences(true_class_id, predictions_, probs_)

    # Get indices of top influential tokens
    top_indices = sorted(range(1, len(influences)), key=lambda x: influences[x], reverse=True)[:num_doc_masks]

    # Map back to token positions
    chosen_positions = [valid_positions[j - 1] for j in top_indices]

    # Create masked version
    masked_input = input_id.clone()
    for pos in chosen_positions:
        masked_input[pos] = mask_token_id

    # Store the result (hashed output) in the cache, using the input key
    cache[input_key] = (masked_input.cpu().numpy(), chosen_positions)
    return masked_input, chosen_positions

def apply_importance_masks(input_ids, attention_mask, model, true_class_ids, tokenizer, server_url, num_doc_masks=10, mask_token_id=None):
    batch_size = input_ids.size(0)
    masked_input_ids = input_ids.clone()
    mask_positions = []

    for i in range(batch_size):
        masked_input, positions = process_single_document(
            input_ids[i],
            attention_mask[i],
            model,
            true_class_ids[i],
            tokenizer,
            num_doc_masks,
            mask_token_id,
            server_url
        )
    
        mask_positions.append(positions)
    # cache_report()
    return masked_input_ids, mask_positions

def save_lists_to_json(list_names: List[str], lists_to_zip: List[List[Any]], output_json_path: str):
    """
    Takes a list of list names and a list of lists, zips them together,
    creates a list of dictionaries, and saves it to a JSON file.

    Args:
        list_names: A list of strings, where each string is the key for a list's elements
                    in the output JSON dictionary. The order of names should correspond
                    to the order of lists in `lists_to_zip`.
        lists_to_zip: A list of lists to be zipped together. The number of lists
                      must match the number of names in `list_names`.
        output_json_path: The path to the JSON file where the results will be saved.

    Raises:
        ValueError: If the number of list names does not match the number of lists.
    """
    if len(list_names) != len(lists_to_zip):
        raise ValueError("The number of list names must match the number of lists to zip.")

    results: List[Dict[str, Any]] = []
    zipped_data = zip(*lists_to_zip)

    for item in zipped_data:
        result_dict: Dict[str, Any] = {}
        for i, name in enumerate(list_names):
            result_dict[name] = item[i]
        results.append(result_dict)

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to: {output_json_path}")

# def sample_multiple_without_replacement(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
#     """
#     Samples multiple tokens without replacement for each position in the logits,
#     keeping the samples for each original token grouped.

#     Args:
#         logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
#                 for each token position in the batch.
#         num_samples: The number of unique tokens to sample without replacement for each
#                      token position.

#     Returns:
#         A tensor of shape (batch_size, num_tokens, num_samples) containing the indices
#         of the sampled tokens.
#     """
#     batch_size, num_tokens, vocab_size = logits.shape
#     all_sampled_tokens = []

#     for i in range(num_tokens):
#         probs = torch.softmax(logits[:, i, :], dim=-1)
#         # Sample without replacement for each batch element
#         indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
#         all_sampled_tokens.append(indices)

#     # Stack the sampled tokens to create the desired shape
#     stacked_sampled_tokens = torch.stack(all_sampled_tokens, dim=1)
#     return stacked_sampled_tokens
    

def get_top_k_indices(logits: torch.Tensor, attacked_positions: List[List[int]], k: int) -> torch.Tensor:
    """
    Gets the top k token indices from the logits for the specified attacked positions.

    Args:
        logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
                for each token position in the batch.
        attacked_positions: A list of lists, where each inner list contains the indices
                            of the tokens to be attacked for a given batch sample.
        k: The number of top token indices to retrieve for each attacked position.

    Returns:
        A tensor of shape (batch_size, max_num_attacked, k) containing the indices
        of the top k tokens for each attacked position. The tensor is padded with
        -1 for positions that were not attacked.
    """
    batch_size, num_tokens, vocab_size = logits.shape
    max_attacked = max(len(positions) for positions in attacked_positions) if attacked_positions else 0
    all_top_k_indices = torch.full((batch_size, max_attacked, k), -1, dtype=torch.long, device=logits.device)

    for b in range(batch_size):
        attack_indices = attacked_positions[b]
        for i, pos in enumerate(attack_indices):
            if pos < num_tokens:  # Ensure the attacked position is within the bounds
                topk_values, topk_indices = torch.topk(logits[b, pos, :], k=k, dim=-1)
                all_top_k_indices[b, i, :] = topk_indices

    return all_top_k_indices

# def get_top_k_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
#     """
#     Gets the top k token indices from the logits for each position,
#     selecting from the top 100 candidates and then randomly choosing 5.

#     Args:
#         logits: A tensor of shape (batch_size, num_tokens, vocab_size) representing the logits
#                 for each token position in the batch.
#         k: The number of token indices to retrieve (in this case, 5).

#     Returns:
#         A tensor of shape (batch_size, num_tokens, k) containing the indices
#         of the randomly selected top k tokens.
#     """
#     batch_size, num_tokens, vocab_size = logits.shape
#     all_top_k_indices = []

#     top_n = 100  # Number of candidates to consider

#     for i in range(num_tokens):
#         # Get the top n values and indices for the current token position
#         top_n_values, top_n_indices = torch.topk(logits[:, i, :], k=top_n, dim=-1)

#         # Ensure random_indices is on the same device as top_n_indices
#         random_indices = torch.randint(low=0, high=top_n, size=(batch_size, k), device=top_n_indices.device)
        
#         # Use gather to get the actual indices
#         top_k_indices = torch.gather(top_n_indices, dim=-1, index=random_indices)
#         all_top_k_indices.append(top_k_indices)

#     # Stack the top k indices to create the desired shape
#     stacked_top_k_indices = torch.stack(all_top_k_indices, dim=1)
#     return stacked_top_k_indices

def generate_token_sample_combinations_batched(sampled_tokens: torch.Tensor, sub_batch_size: int) -> torch.Tensor:
    """
    Generates combinations in smaller sub-batches to manage memory.

    Args:
        sampled_tokens: A tensor of shape (batch_size, num_tokens, num_samples).
        sub_batch_size: The number of documents to process for combinations at a time.

    Returns:
        A tensor of shape (batch_size, num_samples ** num_tokens, num_tokens)
        containing all combinations, processed in batches.
    """
    batch_size, num_tokens, num_samples = sampled_tokens.shape
    all_combinations = []

    for i in range(0, batch_size, sub_batch_size):
        sub_batch = sampled_tokens[i:i + sub_batch_size]
        combinations_sub_batch = []
        for b in range(sub_batch.shape[0]):
            token_samples = sub_batch[b].tolist()
            combinations = list(product(*token_samples))
            combinations_tensor = torch.tensor(combinations, dtype=torch.long)
            combinations_sub_batch.append(combinations_tensor)

        if combinations_sub_batch:
            all_combinations.append(torch.stack(combinations_sub_batch))

    if all_combinations:
        return torch.cat(all_combinations, dim=0)
    else:
        return torch.empty(0, num_samples ** num_tokens, num_tokens, dtype=torch.long)

def generate_candidate_combinations(input_ids: torch.Tensor, attention_mask, sampled_tokens: torch.Tensor, attacked_positions: List[List[int]], original_label: int, tokenizer, server_url) -> List[str]:
    """
    Generates and evaluates adversarial text candidates greedily by replacing token IDs.

    Args:
        input_ids: The original input token IDs (shape: [1, num_tokens]).
        sampled_tokens: The top-k sampled token IDs for each attacked position
                        (shape: [1, max_num_attacked, num_candidates]).
        attacked_positions: A list of lists, where each inner list contains the
                            indices of attacked tokens for a batch sample.
                            In this function (single example processing), we use
                            attacked_positions[0] (List[int]).
        original_label: The true label of the original input (int).
        get_raw_logits_func: The function used to get raw logits and predictions.
        tokenizer: The tokenizer object.
        max_queries: The maximum number of queries allowed (int).

    Returns:
        A list containing the first successful adversarial text candidate found (str),
        or an empty list if none found within the query limit.
    """
    

    atk_succ = False
    src_doc = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    _, src_pred, src_prob = get_raw_logits.process_file(data=[src_doc], server_url=server_url)
    
    gen_doc = src_doc
    src_pred_label = src_pred[0]
    src_pred_prob = src_prob[0]
    adv_pred_label = src_pred[0]
    adv_pred_prob = src_prob[0]
    worst_prob = src_pred_prob
    queries_used = 0
    queries_used = 0
    nums_pert_toks = 0
    src_len = torch.sum(attention_mask).item()
    pert_rate = nums_pert_toks / src_len

    if src_pred[0] != original_label:
        atk_succ = True
        return atk_succ, src_doc, gen_doc, original_label, src_pred_label, src_pred_prob,\
              adv_pred_label, adv_pred_prob, worst_prob, \
                queries_used, nums_pert_toks, src_len, pert_rate

    nums_of_batch = sampled_tokens.shape[0]
    nums_atk_toks = sampled_tokens.shape[1]
    nums_tok_candidates = sampled_tokens.shape[2]
    
    input_id_curr = input_ids[0].clone().cpu().numpy()
    worst_prob = 1.0

    for atk_idx in range(nums_atk_toks):
        pos = attacked_positions[0][atk_idx]
        candidate_token_ids = sampled_tokens[0, atk_idx, :].tolist()

        for candidate_id in candidate_token_ids:
            temp_input_ids = list(input_id_curr)
            temp_input_ids[pos] = candidate_id
            candidate_text = tokenizer.decode(temp_input_ids, skip_special_tokens=True)

            _, predictions, prob = get_raw_logits.process_file(data=[candidate_text], server_url=server_url)
            queries_used += 1

            if predictions[0] != original_label:
                atk_succ = True
                src_doc = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                gen_doc = candidate_text
                src_pred_label = src_pred[0]
                src_pred_prob = src_prob[0]
                adv_pred_label = predictions[0]
                adv_pred_prob = 1 - prob[0]
                queries_used = 0
                nums_pert_toks = atk_idx + 1
                # src_len = torch.sum(attention_mask).item()
                pert_rate = nums_pert_toks / src_len

                return atk_succ, src_doc, gen_doc, original_label, src_pred_label, src_pred_prob,\
              adv_pred_label, adv_pred_prob, worst_prob, \
                queries_used, nums_pert_toks, src_len, pert_rate
            
            elif prob[0] < worst_prob:
                worst_prob = prob[0]
                input_id_curr = temp_input_ids

    return atk_succ, src_doc, gen_doc, original_label, src_pred_label, src_pred_prob,\
                adv_pred_label, adv_pred_prob, worst_prob, \
                    queries_used, nums_pert_toks, src_len, pert_rate


def main(args):
    atker_path = args.atker_path
    target_path = args.target_path
    len_doc_max = args.len_doc_max
    num_doc_masks = args.num_doc_masks
    save_to_path = args.save_to_path
    samples_per_tok = args.samples_per_tok
    atk_json_log = args.atk_json_log
    server_url = args.server_url
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # best_gpu = GPUtil.getFirstAvailable(order='memoryFree', maxLoad=0.5, maxMemory=0.5)[0]
    # torch.cuda.set_device(best_gpu)
    # device = torch.device(f"cuda:{best_gpu}")
    # print(f"Using GPU {best_gpu}")

    USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")
    # with tf.device(f"/GPU:{best_gpu}"):
    #     USE = hub.load("https://kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/1")

    tokenizer = AutoTokenizer.from_pretrained(atker_path, max_length=len_doc_max)
    vocab_size = tokenizer.vocab_size
    print(f'Attacker Base: {atker_path}')
    print(f"Vocabulary Size: {vocab_size}")

    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    unk_token_id = tokenizer.unk_token_id

    _, _, evaluation_dataloader = build_datasets(tokenizer=tokenizer, num_doc_masks=num_doc_masks, max_len=len_doc_max, seed=42)
    
    # Initialize attacker
    config = BertConfig.from_pretrained(atker_path, output_hidden_states=True)
    model = BertForMaskedLM.from_pretrained(atker_path, config=config).to(device)
    
    # Load the trained model weights
    state_dict = torch.load(save_to_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)  # Load the weights

    # Freeze all layers.  This is evaluation, so typically we don't change model weights.
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.eval() # Set to eval mode
    
    def attack(evaluation_dataloader=evaluation_dataloader, attacker=model, eval_interval=100):
        model.eval()
        orig_acc_metric = Accuracy(task="binary").to(device)
        atk_acc_metric = Accuracy(task="binary").to(device)
        all_results = []
        all_predicted_labels = []
        all_true_labels = []
        all_source_predicted_probs = []
        all_generated_predicted_probs = []
        all_source_documents = []
        all_generated_documents = []
        queries = []
        USEs = []
        pertubations = []
        all_attacked_positions = []

        # Initialize the lists here
        all_source_documents = []
        all_generated_documents = []
        all_true_labels = []
        all_source_predicted_labels = []
        all_source_predicted_probs = []
        all_gen_predicted_labels = []
        all_gen_predicted_probs = []
        all_worst_probs = []
        all_quries = []
        all_nums_pert_toks = []
        all_src_len = []
        all_pert_rate = []
        with torch.no_grad():
            total_batches = len(evaluation_dataloader)
            bar = tqdm(total=total_batches, desc="Evaluating", unit="batch")            
            for batch_idx, batch in enumerate(evaluation_dataloader, start=1):
                curr_queries_per_doc = 0

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                src_doc = tokenizer.decode(input_ids[0],skip_special_tokens=True)
                prompt_, predictions_, probs_ = get_raw_logits.process_file(data=[src_doc], server_url=server_url) 

                masked_input_ids, attacked_positions = apply_importance_masks(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    model=attacker,  # Use the attacker model
                    true_class_ids=labels,
                    tokenizer=tokenizer,
                    server_url=server_url,
                    num_doc_masks=num_doc_masks, # Use num_doc_masks
                    mask_token_id=mask_token_id
                )

                all_attacked_positions.append(attacked_positions) # Store
                if not attacked_positions[0]: # Check the first (and only) element.
                    print('skip document {batch_idx} because it is an empty document.')
                    continue

                logits = model(masked_input_ids, attention_mask).logits
                sampled_tokens = get_top_k_indices(logits=logits, attacked_positions=attacked_positions, k=samples_per_tok) # [batch_size, num_attacked_positions, samples_per_tok]

                atk_succ, src_doc, gen_doc, original_label, src_pred_label, src_pred_prob,\
                adv_pred_label, adv_pred_prob, worst_prob, \
                    queries_used, nums_pert_toks, src_len, pert_rate = generate_candidate_combinations(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sampled_tokens=sampled_tokens,
                    attacked_positions=attacked_positions,
                    original_label = labels.item(),
                    tokenizer = tokenizer,
                    server_url=server_url)
                
                all_source_documents.append(src_doc)
                all_generated_documents.append(gen_doc)
                all_true_labels.append(original_label) 
                all_source_predicted_labels.append(src_pred_label)
                all_source_predicted_probs.append(src_pred_prob)
                all_gen_predicted_labels.append(adv_pred_label)
                all_gen_predicted_probs.append(adv_pred_prob)
                all_worst_probs.append(worst_prob)
                all_quries.append(queries_used)
                all_nums_pert_toks.append(nums_pert_toks)
                all_src_len.append(src_len)
                all_pert_rate.append(pert_rate)
                
                all_true_labels_tensor = torch.tensor(all_true_labels).to(device)
                all_source_predicted_labels_tensor = torch.tensor(all_source_predicted_labels).to(device)
                all_gen_predicted_labels_tensor = torch.tensor(all_gen_predicted_labels).to(device)
                
                orig_acc_metric.update(all_source_predicted_labels_tensor, all_true_labels_tensor)
                atk_acc_metric.update(all_gen_predicted_labels_tensor, all_true_labels_tensor)
                original_accuracy = orig_acc_metric.compute()
                current_accuracy = atk_acc_metric.compute()
    
                USEs.append(getUSEcosSimilarity([all_source_documents[-1]], [all_generated_documents[-1]], USE)[0])

                bar.update(1)
                bar.set_postfix({
                    "avg_ori_acc": f"{original_accuracy:.4f}",
                    "avg_atk_acc": f"{current_accuracy:.4f}",
                    "avg_queries": f"{np.mean(all_quries):.4f}",
                    "avg_pert": f"{np.mean(all_pert_rate) * 100:.2f}",
                    "avg_USE": f"{np.mean(USEs):.4f}",
                    "cur_queries": f"{all_quries[-1]:.4f}",
                    "cur_pert": f"{all_pert_rate[-1] * 100:.2f}",
                    "cur_USE": f"{USEs[-1]:.4f}"
                })

                # if batch_idx % eval_interval == 0:
                #     print(f"\n--- Step {batch_idx} ---")
                #     print("Source Documents (last few):", all_source_documents[-1])
                #     print("Generated Documents (last few):", all_generated_documents[-1])
                #     print("True Labels (last few):", all_true_labels[-1])
                #     print("Predicted Labels (last few):", all_gen_predicted_labels[-1])
                #     print("Source Predicted Probabilities:", round(all_source_predicted_probs[-1],4))
                #     print("Generated Predicted Probabilities:", round(all_generated_predicted_probs[-1],4))
                
                list_names = [
                    "src_doc", "adv_doc", "true_label", "src_pred_label", "src_pred_prob",
                    "adv_pred_label", "adv_pred_prob", "worst_prob", "queries_used",
                    "num_perturbed_tokens", "src_length", "perturbation_rate", "USEs"]
                    
                lists_to_zip = [
                    all_source_documents,
                    all_generated_documents,
                    all_true_labels,
                    all_source_predicted_labels,
                    all_source_predicted_probs,
                    all_gen_predicted_labels,
                    all_gen_predicted_probs,
                    all_worst_probs,
                    all_quries,
                    all_nums_pert_toks,
                    all_src_len,
                    all_pert_rate,
                    USEs]
                save_lists_to_json(list_names=list_names, lists_to_zip=lists_to_zip, output_json_path=atk_json_log)
            bar.close()
        
    attack(evaluation_dataloader=evaluation_dataloader, attacker=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--atker_path', type=str, required=True, help='path folder to load attacker models')
    parser.add_argument('--target_path', type=str, required=True, help='target model path')
    parser.add_argument('--save_to_path', type=str, required=True, help='target model path')
    parser.add_argument('--len_doc_max', type=int, default=512, help='max length of document')  # Default value set to 512
    parser.add_argument('--num_doc_masks', type=int, default=10, help='')  
    parser.add_argument('--samples_per_tok', type=int, default=10, help='')  # Default value set to 512
    parser.add_argument('--atk_json_log', type=str, default=10, help='') 
    parser.add_argument('--server_url', type=str, default="http://localhost:8000/v1", help='8000 for llama guard 3 1B, 8001 for 8B')  # Default value set to 512

    # args = argparse.Namespace(
    #         atker_path='bert-base-uncased', # Example path
    #         target_path='temp',
    #         len_doc_max=512,
    #         num_doc_masks=10, # numbers of attacked tokens maximumlly allowed
    #         save_to_path='/usa/taikun/07_transencoder/1training/llama-guard-attacker/attacker_llama-guard_4_5100_0.6450.pth',
    #         samples_per_tok=5,
    #         # atk_json_log = '/usa/taikun/07_transencoder/attack-genai/atk_greedy_topk_doc_log.json')
    #         atk_json_log = '/usa/taikun/07_transencoder/attack-genai/temp.json')
        
    args = parser.parse_args()    
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    main(args)
