import sys
from pathlib import Path

import torch
import torch.distributed as dist
import os

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from dataset.packing.packed_dataset import PackedDataset
from datasets import load_dataset, Dataset, load_from_disk
from dataset.prompter import Prompter
from transformers import AutoTokenizer
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, tokenizer, data_path, prompter, train_on_inputs=False):
    if 'lamini' in data_path.lower():
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            None,
            data_point["response"],
        )
    elif 'alpaca' in data_path.lower():
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
    else:
        raise NotImplementedError

    tokenized_full_prompt = tokenize(full_prompt, tokenizer=tokenizer)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
        )
        tokenized_user_prompt = tokenize(
            user_prompt, tokenizer=tokenizer
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # if add_eos_token:
        if tokenized_user_prompt["input_ids"][-1] == tokenizer.eos_token_id:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def load_LaMini_train(data_path, tokenizer, debug=False):

    prompter = Prompter("alpaca")
    dataset = load_dataset(data_path)["train"]
    if debug:
        dataset = dataset.select(range(100))
    dataset = dataset.shuffle().map(
                lambda data_point: generate_and_tokenize_prompt(
                    data_point=data_point, 
                    tokenizer=tokenizer,
                    data_path=data_path, 
                    prompter=prompter, 
                    train_on_inputs=False,
                )
            )
    # dataset = dataset.remove_columns(['instruction', 'response', 'instruction_source'])
    # dataset.save_to_disk(f"out/cache/Lamini-llama3.2")

    return dataset

def load_alpaca_train(data_path, tokenizer, max_length, cache=False, debug=False):
    if not cache:
        prompter = Prompter("alpaca")
        dataset = load_dataset(data_path)["train"]
        if debug:
            dataset = dataset.select(range(100))
        dataset = dataset.shuffle().map(
                    lambda data_point: generate_and_tokenize_prompt(
                        data_point=data_point, 
                        tokenizer=tokenizer,
                        max_length=max_length,
                        data_path=data_path, 
                        prompter=prompter, 
                        train_on_inputs=False,
                        add_eos_token=True
                    )
                )
        dataset = dataset.remove_columns(['instruction', 'input', 'output'])
        dataset.save_to_disk(f"/home/zhuhr/code/llm_prune/out/cache/alpaca_llama32_{max_length}")
    else:
        dataset = Dataset.load_from_disk(f"/home/zhuhr/code/llm_prune/out/cache/alpaca_llama32_{max_length}")
    return dataset

def truncate_dataset(dataset, max_length):
    def truncate_features(example):
        example["input_ids"] = example["input_ids"][:max_length]
        example["attention_mask"] = example["attention_mask"][:max_length]
        example["labels"] = example["labels"][:max_length]
        return example
    dataset = dataset.map(truncate_features)
    return dataset
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/public/model_weight/Llama3.2/Llama-3.2-3B")
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    max_length=5120 #6144 5120 4096 3072

    # dataset = load_from_disk("out/cache/Lamini-llama3.2-clean")
    # dataset = truncate_dataset(dataset, max_length)
    save_path = f"out/cache/Lamini-llama3.2-clean-{max_length}"
    # dataset.save_to_disk(save_path)

    dataset = load_from_disk(save_path)
    dataset = PackedDataset(dataset, tokenizer=tokenizer, max_input_length=max_length, pack_length=max_length)
    
    dataset.save(os.path.join(save_path,"data_dict.json"))
    print(os.path.join(save_path,"data_dict.json"))
