
def packed_ft_llama_Lamini_config():
    config = {
        "project_name": "packed_ft-llama3.2_Lamini",
        "dataset": {
            "name": "out/cache/Lamini-llama3.2-clean-4096",
            "sub_name": "fineweb-edu-dedup",
            "split": "train",
            "seed": 42,
            "dataset_text_field": "text",
            "packing": True
        },
        "models": {
            "teacher": "Llama3.2-1B_path",
            "student": "pruned_model_path"
        },
        "tokenizer": {
            "max_length": 4096
        },
        "training": {
            "output_dir": "./out",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "save_steps": 500,
            "logging_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "warmup_ratio": 0,
            "lr_scheduler_type": "cosine",
            "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
            "fp16": False,
            "bf16": True,
            "remove_unused_columns": False,
            "optim": "adamw_anyprecision",
            "optim_args" : "momentum_dtype=bfloat16,variance_dtype=bfloat16",
            "dataloader_num_workers": 0,
        },
        "model_config": {
            "use_flash_attention": True
        }
    }
    return config
def packed_ft_llama_Lamini_config_lora():
    config = {
        "project_name": " packed_ft_llama_Lamini",
        "dataset": {
            "name": "out/cache/Lamini-llama3.2-clean-4096",
            "sub_name": "fineweb-edu-dedup",
            "split": "train",
            "seed": 42,
            "dataset_text_field": "text",
            "packing": True
        },
        "models": {
            "teacher": "Llama-3.2-1B_path",
            "student": "pruned_model_path"
        },
        "tokenizer": {
            "max_length": 4096
        },
        "training": {
            "output_dir": "./out",
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "save_steps": 500,
            "logging_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "warmup_ratio": 0,
            "lr_scheduler_type": "cosine",
            "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
            "fp16": False,
            "bf16": True,
            "remove_unused_columns": False,
            "optim": "adamw_anyprecision",
            "optim_args" : "momentum_dtype=bfloat16,variance_dtype=bfloat16",
            "dataloader_num_workers": 0,
        },
        "model_config": {
            "use_flash_attention": True
        },
        "lora": {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
        },
    }
    return config
