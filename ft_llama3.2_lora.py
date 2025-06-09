import argparse
import os
from cfg.distill_llama import packed_ft_llama_Lamini_config_lora
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, prepare_model_for_kbit_training
from dataset.packing.monkey_patch_packing import monkey_patch_packing_for_model
from dataset.packing.packed_dataset import PackedDataset
from module.trainer import PackedSFTTrainer
from utils import get_output_dir
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main(config):
    
    # Set up environment
    os.environ['WANDB_PROJECT'] = config["project_name"]
    accelerator = Accelerator()
    device = accelerator.device
    config["training"]["output_dir"] = get_output_dir(config)
    config["training"]["logging_dir"] = os.path.join(config["training"]["output_dir"], "logs")



    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset_path = config["dataset"]["name"]
    dataset = load_from_disk(dataset_path)
    dataset = PackedDataset(dataset, tokenizer=tokenizer, max_input_length=config["tokenizer"]["max_length"], pack_length=config["tokenizer"]["max_length"], cache=f"{dataset_path}/data_dict.json")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding="longest", max_length=config["tokenizer"]["max_length"])    
    print("Dataset preparation complete. Loading models...")

    # Load models with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    monkey_patch_packing_for_model(config["models"]["student"])
    model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)
    
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config["lora"]["lora_r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["lora_target_modules"].split(","),
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        peft_type="LORA"
    )

    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters() 
    

    model.add_adapter(lora_config)
    model.enable_adapters()
    model.config.use_cache = False
    if accelerator.is_main_process:
        os.makedirs(config["training"]["output_dir"], exist_ok=True)
        os.makedirs(config["training"]["logging_dir"], exist_ok=True)
    if config["training"]["resume_from_checkpoint"] is not None:
        dataset.load_dataset_states(config["training"]["output_dir"])
    # Training arguments
    training_arguments = TrainingArguments(**config["training"])
    # Create the custom SFT Trainer
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
    if accelerator.is_main_process:
        print(config["models"]["student"], "ft is start!")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    model.save_pretrained(config["training"]["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--student_model', type=str, default=None)
    parser.add_argument('--ft_parts', type=str, default="all")
    args = parser.parse_args()
    config = packed_ft_llama_Lamini_config_lora()
    if args.student_model is not None:
        config["models"]["student"] = args.student_model
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.ft_parts == "mlp":
        config["lora"]["lora_target_modules"] = "gate_proj,down_proj,up_proj"

    main(config)