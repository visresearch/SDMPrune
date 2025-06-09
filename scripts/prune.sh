#!/bin/bash  
#SBATCH -o out/slurm/prune.%j.out ##作业的输出信息文件  
#SBATCH -J prune ##作业名  
#SBATCH -p 4090D
#SBATCH --nodes=1 ##申请1个节点 
#SBATCH --mem=125000M 
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=16


source ~/.bashrc

cd /home/zhuhr/code/llm_prune
conda activate llm_prune

# lora ft 20% pruning rate
# seed=0
# mlp_ratio=2.741
# alpha=0.4
# t=0.2

# lora ft 30% pruning rate
# seed=0
# mlp_ratio=2.161
# alpha=0.42
# t=0.5

# lora ft 40% pruning rate
# seed=0
# mlp_ratio=1.548
# alpha=0.1
# t=0.2



python prune/distill_prune.py --t $t --seed $seed --alpha $alpha --mlp_ratio $mlp_ratio
ckpt="out/model/distill/kl_Llama-3.2-1B_distill_${mlp_ratio}_seed${seed}_t${t}_alpha${alpha}"
bash scripts/eval_baseline.sh "$ckpt"
python utils/show_res.py --path "$ckpt"
