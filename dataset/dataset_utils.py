


import copy
import sys
import torch


class CustomDataset:
    def __init__(self, tokenizer, dataset, max_length=1024):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length
        )
        if len(encoding["input_ids"])<self.max_length:
            encoding["input_ids"].append(self.tokenizer.eos_token_id)
            encoding["attention_mask"].append(1)
        labels = copy.copy(encoding['input_ids'])
        labels[0] = -100
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels,
        }

    def __len__(self):
        return len(self.dataset)


class PackedCustomDataset:
    def __init__(self, dataset, tokenizer, max_length=1024, max_size=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.total_length = len(dataset)
        self.used_length = 0
        self.max_size = 10
    def __getitem__(self, idx):
        if self.used_length >= self.total_length:
            return None
        else:
            n_data_point={}
            for i in range(self.max_size):
                item = self.dataset[self.used_length]
                encoding = self.tokenizer(
                    item['text'],
                )
                if encoding["input_ids"][-1] != self.tokenizer.eos_token_id:
                    encoding["input_ids"].append(self.tokenizer.eos_token_id)
                    encoding["attention_mask"].append(1)
                encoding["attention_mask"] = [ x+i for x in encoding["attention_mask"]]
                labels = copy.copy(encoding['input_ids'])
                labels[0] = -100

                if i == 0 and len(encoding["input_ids"]) >=self.max_length:
                    encoding["input_ids"] = encoding["input_ids"][:self.max_length]
                    encoding["attention_mask"] = encoding["attention_mask"][:self.max_length]
                    labels = labels[:self.max_length]
                    self.used_length += 1
                    return {
                        'input_ids': encoding['input_ids'],
                        'attention_mask': encoding['attention_mask'],
                        'labels': labels,
                    }
                
                if i == 0:
                    n_data_point = {
                        'input_ids': encoding['input_ids'],
                        'attention_mask': encoding['attention_mask'],
                        'labels': labels,
                    }
                    self.used_length += 1
                else:
                    if len(n_data_point["input_ids"]) + len(encoding["input_ids"]) == self.max_length:
                        n_data_point = {
                            'input_ids': n_data_point['input_ids'] + encoding["input_ids"],
                            'attention_mask': n_data_point['attention_mask'] + encoding['attention_mask'],
                            'labels': n_data_point['labels'] + labels,
                        }
                        self.used_length += 1
                        return n_data_point
                    elif len(n_data_point["input_ids"]) + len(encoding["input_ids"]) < self.max_length:
                        n_data_point = {
                            'input_ids': n_data_point['input_ids'] + encoding["input_ids"],
                            'attention_mask': n_data_point['attention_mask'] + encoding['attention_mask'],
                            'labels': n_data_point['labels'] + labels,
                        }
                        self.used_length += 1
                    elif len(n_data_point["input_ids"]) + len(encoding["input_ids"]) > self.max_length:
                        return n_data_point
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("/public/model_weight/Llama3.2/Llama-3.2-1B-Instruct")
    dataset = load_dataset("/public/dataset/smollm-corpus", "fineweb-edu-dedup",split="train", num_proc=8)
    dataset = dataset.select(range(1000))
    dataset = PackedCustomDataset(dataset, tokenizer, 4096)
    for i in range(len(dataset)):
        item = dataset[i]
        print(len(item["attention_mask"]))
        print(item["attention_mask"])
        assert len(item["input_ids"]) == len(item["attention_mask"]) == len(item["labels"])