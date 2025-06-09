import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import torch

import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
import torch.nn as nn
import sys
from tqdm import tqdm
import time
import torch.nn.functional as F

def prune_given_indices(in_linear: nn.Linear, out_linear: nn.Linear, gate_linear: nn.Linear, idxs_select: torch.Tensor=None):
    in_weight = in_linear.weight.data
    in_bias = in_linear.bias.data if in_linear.bias is not None else None
    out_weight = out_linear.weight.data
    out_bias = out_linear.bias.data if out_linear.bias is not None else None
    gate_weight = gate_linear.weight.data
    gate_bias = gate_linear.bias.data if gate_linear.bias is not None else None
    
    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_select]
    in_bias_prune = in_bias[idxs_select] \
        if in_bias is not None else None
    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )
    in_linear_prune.weight.data = in_weight_prune
    if in_bias is not None:
        in_linear_prune.bias.data = in_bias_prune
    # --------------- gate linear -----------------
    gate_weight_prune = gate_weight[idxs_select]
    gate_bias_prune = gate_bias[idxs_select] \
        if gate_bias is not None else None
    gate_linear_prune = nn.Linear(
        in_features=gate_weight_prune.shape[1], 
        out_features=gate_weight_prune.shape[0],
        bias=gate_bias_prune is not None
    )
    gate_linear_prune.weight.data = gate_weight_prune
    if gate_bias is not None:
        gate_linear_prune.bias.data = gate_bias_prune
    # --------------- output linear ---------------
    out_weight_prune = out_weight[:, idxs_select]
    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )
    out_linear_prune.weight.data = out_weight_prune
    if out_bias is not None:
        out_linear_prune.bias.data = out_bias
    return in_linear_prune, out_linear_prune, gate_linear_prune
def tokenize(data_proint, tokenizer):
    example = tokenizer(data_proint["text"])
    return example
def gradient_abs_hook(grad):
    return grad.abs()

def get_importance_taylor(path, seed=None):
    seq_len = 1024
    dataset = load_dataset('/public/dataset/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    if seed:
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = dataset.shuffle()
    tokenizer = AutoTokenizer.from_pretrained(path)
    filtered_samples = []
    for example in dataset:
        inputs = tokenizer(example['text'], truncation=False, padding=False)
        if len(inputs['input_ids']) > seq_len:
            i = random.randint(0, len(inputs['input_ids']) - seq_len - 1)
            j = i + seq_len
            inputs['input_ids'] = inputs['input_ids'][i:j]
            inputs['attention_mask'] = inputs['attention_mask'][i:j]
            filtered_samples.append(inputs)
        if len(filtered_samples) >= 1024:
            break
    dataset = Dataset.from_list(filtered_samples)
    print(dataset)
    model = AutoModelForCausalLM.from_pretrained(path)
    for p in model.parameters():
        p.requires_grad = False
    for idx in range(len(model.model.layers)):
        for p in model.model.layers[idx].mlp.down_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.up_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.gate_proj.parameters():
            p.requires_grad = True
    model = model.to(torch.bfloat16).cuda()
    
    print("forward!")
    importance = [torch.zeros_like(layer.mlp.gate_proj.weight.data) for layer in model.model.layers]
    for example in tqdm(dataset):
        input_ids = torch.tensor(example["input_ids"], dtype=torch.int).unsqueeze(0)
        labels = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
        # labels[:,:-1] = -100
        input_ids, labels = input_ids.cuda(), labels.cuda()
        loss = model(input_ids=input_ids, labels=labels)[0]
        loss.backward()
    
        for idx in range(len(model.model.layers)):
            # prevent from out of precision range
            down_proj_importance = model.model.layers[idx].mlp.down_proj.weight.grad.T * 100 * model.model.layers[idx].mlp.down_proj.weight.data.T
            up_proj_importance = model.model.layers[idx].mlp.up_proj.weight.grad * 100 * model.model.layers[idx].mlp.up_proj.weight.data
            gate_proj_importance = model.model.layers[idx].mlp.gate_proj.weight.grad * 100 * model.model.layers[idx].mlp.gate_proj.weight.data
            importance[idx] += torch.abs(down_proj_importance + up_proj_importance + gate_proj_importance)
        
        for param in model.parameters():
            param.grad = None
 
    res = {"importance": importance}
    return res   
def prune(importance, origin_path, save_path, mlp_ratio):
    model = AutoModelForCausalLM.from_pretrained(origin_path)
    model = model.to(torch.bfloat16).cuda()
    importance = importance["importance"]
    
    importance = [torch.mean(item, dim=-1) for item in importance]
    
    for idx in range(len(model.model.layers)):
        importance[idx] = importance[idx].cuda()
        dim = model.model.layers[idx].mlp.down_proj.out_features
        _, indices = torch.topk(importance[idx], k=int(dim * mlp_ratio), dim=0, largest=True)
        indices, _ = torch.sort(indices)
        model.model.layers[idx].mlp.up_proj, model.model.layers[idx].mlp.down_proj, \
            model.model.layers[idx].mlp.gate_proj = \
        prune_given_indices(model.model.layers[idx].mlp.up_proj, model.model.layers[idx].mlp.down_proj, model.model.layers[idx].mlp.gate_proj, indices)
    
    model.config.intermediate_size = int(mlp_ratio * model.config.hidden_size)
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(origin_path)
    tokenizer.save_pretrained(save_path)

def get_importance_kl(origin_path, path, seed=None, t=1, alpha=1):
    seq_len = 1024
    dataset = load_dataset('/public/dataset/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    if seed:
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = dataset.shuffle()
    tokenizer = AutoTokenizer.from_pretrained(origin_path)
    filtered_samples = []
    for example in dataset:
        inputs = tokenizer(example['text'], truncation=False, padding=False)
        if len(inputs['input_ids']) > seq_len:
            i = random.randint(0, len(inputs['input_ids']) - seq_len - 1)
            j = i + seq_len
            inputs['input_ids'] = inputs['input_ids'][i:j]
            inputs['attention_mask'] = inputs['attention_mask'][i:j]
            filtered_samples.append(inputs)
        if len(filtered_samples) >= 1024:
            break
    dataset = Dataset.from_list(filtered_samples)
    print(dataset)
    model = AutoModelForCausalLM.from_pretrained(path)
    teacher_model = AutoModelForCausalLM.from_pretrained(origin_path)
    for p in model.parameters():
        p.requires_grad = False
    for p in teacher_model.parameters():
        p.requires_grad = False
    for idx in range(len(model.model.layers)):
        for p in model.model.layers[idx].mlp.down_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.up_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.gate_proj.parameters():
            p.requires_grad = True
    model = model.to(torch.bfloat16).cuda()
    teacher_model = teacher_model.to(torch.bfloat16).cuda()
    print("forward!")
    importance = [torch.zeros_like(layer.mlp.gate_proj.weight.data) for layer in model.model.layers]
    for example in tqdm(dataset):
        input_ids = torch.tensor(example["input_ids"], dtype=torch.int).unsqueeze(0)
        labels = torch.tensor(example["input_ids"], dtype=torch.long).unsqueeze(0)
        # labels[:,:-1] = -100
        input_ids, labels = input_ids.cuda(), labels.cuda()
        teacher_logits = teacher_model(input_ids=input_ids)[0]
        out = model(input_ids=input_ids, labels=labels)
        ce_loss = out[0]
        logits = out[1]
        loss_kd = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(teacher_logits/t, dim=-1),
            reduction='batchmean'
        )/seq_len
        # loss_kd = F.mse_loss(F.softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1))
        loss = (1-alpha) * ce_loss + alpha * loss_kd
        # loss = loss_kd

        loss.backward()
    
        for idx in range(len(model.model.layers)):
            # prevent from out of precision range
            down_proj_importance = model.model.layers[idx].mlp.down_proj.weight.grad.T * 100 * model.model.layers[idx].mlp.down_proj.weight.data.T
            up_proj_importance = model.model.layers[idx].mlp.up_proj.weight.grad * 100 * model.model.layers[idx].mlp.up_proj.weight.data
            gate_proj_importance = model.model.layers[idx].mlp.gate_proj.weight.grad * 100 * model.model.layers[idx].mlp.gate_proj.weight.data
            importance[idx] += torch.abs(down_proj_importance + up_proj_importance + gate_proj_importance)
        
        for param in model.parameters():
            param.grad = None
 
    res = {"importance": importance}
    return res   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SDMPrune API")
    parser.add_argument("--t", type=float, required=True, help="t")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument("--alpha", type=float, required=True, help="alpha")
    parser.add_argument("--mlp_ratio", type=float, required=True, help="mlp_ratio")
    args = parser.parse_args()

    origin_path = "/public/model_weight/Llama3.2/Llama-3.2-1B"
    seed = args.seed

    t=args.t
    alpha = args.alpha
    mlp_ratio = 3.9
    taylor_importance = get_importance_taylor(origin_path, seed=seed)
    # taylor start
    taylor_save_path = f"out/model/distill/taylor_Llama-3.2-1B_distill_{mlp_ratio}_seed{seed}_t{t}_alpha{alpha}"
    prune(taylor_importance, origin_path, taylor_save_path, mlp_ratio)

    
    mlp_ratio = args.mlp_ratio # 2.774 20%; 2.161 30% ;1.548 40%
    kl_importance = get_importance_kl(origin_path, taylor_save_path, seed=seed, t=t, alpha=alpha)
    kl_save_path = f"out/model/distill/kl_Llama-3.2-1B_distill_{mlp_ratio}_seed{seed}_t{t}_alpha{alpha}"
    prune(kl_importance, taylor_save_path, kl_save_path, mlp_ratio)


    