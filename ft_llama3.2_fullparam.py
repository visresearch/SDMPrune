import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, LlamaForCausalLM
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq, Trainer
from dataset.packing.monkey_patch_packing import monkey_patch_packing_for_model
from cfg.distill_llama import packed_ft_llama_Lamini_config
from module.trainer import  PackedSFTTrainer
from torch.utils.data import DataLoader
from dataset.packing.monkey_patch_packing import monkey_patch_packing_for_model
from dataset.packing.packed_dataset import PackedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils import get_output_dir


def main(config):
    
    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    accelerator = Accelerator()
    device = accelerator.device
    config["training"]["output_dir"] = get_output_dir(config)
    config["training"]["logging_dir"] = os.path.join(config["training"]["output_dir"], "logs")

    tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_path = config["dataset"]["name"]
    dataset = load_from_disk(dataset_path)
    dataset = PackedDataset(dataset, tokenizer=tokenizer, max_input_length=4096, pack_length=4096, cache=f"{dataset_path}/data_dict.json")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding="longest", max_length=config["tokenizer"]["max_length"])    
    print("Dataset preparation complete. Loading models...")
    # Load models with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    monkey_patch_packing_for_model(config["models"]["student"])
    model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)
    # model = LlamaForCausalLMPacked.from_pretrained(config["models"]["student"], **model_kwargs)
    model.config.use_cache = False
    if accelerator.is_main_process:
        os.makedirs(config["training"]["output_dir"], exist_ok=True)
        os.makedirs(config["training"]["logging_dir"], exist_ok=True)
    if config["training"]["resume_from_checkpoint"] is not None:
        dataset.load_dataset_states(config["training"]["output_dir"])
    # Training arguments
    training_arguments = TrainingArguments(**config["training"])

    trainer = PackedSFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
        args=training_arguments,
    )


    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)
    # Train the model
    if accelerator.is_main_process():
        print(config["models"]["student"], "ft is start!")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(config["training"]["output_dir"])


if __name__ == "__main__":
    config = packed_ft_llama_Lamini_config()
    # config["training"]["output_dir"] = get_output_dir(config)
    main(config)