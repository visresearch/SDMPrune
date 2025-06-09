#!/bin/bash  
#SBATCH -o out/slurm/eval.%j.out ##作业的输出信息文件  
#SBATCH -J eval ##作业名  
#SBATCH -p 4090D
#SBATCH --nodes=1 ##申请1个节点 
#SBATCH --mem=125000M 
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=16
source ~/.bashrc

cd /home/zhuhr/code/llm_prune
conda activate llm_prune


if [ -z "$1" ]; then
    echo "Error: No pruned_model_path provided."
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: No adaptor_path provided."
    exit 1
fi
model_path=$1

ckpt=$2
if [ -e "$ckpt" ]; then
    echo "Evaluating checkpoint: $ckpt"
    output_dir=$(dirname "$ckpt")
    lm_eval --model hf \
        --model_args pretrained=$model_path,peft=$ckpt,trust_remote_code=True,add_bos_token=True \
        --tasks hellaswag,piqa,arc_challenge,arc_easy,openbookqa,boolq,winogrande,truthfulqa_mc2,crows_pairs_english,race,social_iqa \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "$ckpt/results"
else
    echo "Checkpoint not found: $ckpt"
fi
