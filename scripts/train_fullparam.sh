#!/bin/bash
#!/bin/bash  
#SBATCH -o out/slurm/train%j.out ##作业的输出信息文件  
#SBATCH -J prune ##作业名  
#SBATCH -p 4090D
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:4 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=64
source ~/.bashrc

cd /home/zhuhr/code/llm_prune
conda activate llm_prune

accelerate launch --num_processes=4 --num_machines 1 \
--mixed_precision bf16 --dynamo_backend no \
ft_llama3.2_fullparam.py --student_model out/pruned_model_path \
--lr 1e-4


