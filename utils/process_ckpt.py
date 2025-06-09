import os
import shutil

def remove_state_files(root_dir, min_n, max_n, step):
    files_to_delete = ['optimizer.pt', 'rng_state_0.pth', 'rng_state_1.pth', 'rng_state_2.pth', 'rng_state_3.pth', 'scheduler.pt', 'trainer_state.json', 'training_args.bin', "dataset.dp_rank_00","dataset.dp_rank_01","dataset.dp_rank_02","dataset.dp_rank_03"]  # 替换为你要删除的文件名
    for i in range(min_n, max_n, step):
        parent_directory = os.path.join(root_dir, f'checkpoint-{i}')
        for file_name in files_to_delete:
            file_path = os.path.join(parent_directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Deleted: {file_path}')


def copy_tokenizer(parent_directory, min_n, max_n, step):
    # 定义要复制的文件
    files = ['/public/model_weight/Llama3.2/Llama-3.2-1B-Instruct/special_tokens_map.json', '/public/model_weight/Llama3.2/Llama-3.2-1B-Instruct/tokenizer.json', '/public/model_weight/Llama3.2/Llama-3.2-1B-Instruct/tokenizer_config.json']  # 替换为你的文件名或路径


    # 遍历目标目录
    for i in range(min_n, max_n + step, step):
        target_dir = os.path.join(parent_directory, f'checkpoint-{i}')
        if os.path.exists(target_dir):
            for file in files:
                shutil.copy(file, target_dir)
                print(f'Copied {file} to {target_dir}')
        else:
            print(f"{target_dir} not found!")
if __name__ == "__main__":
    root_dir = 'out/packed_pretrain-llama3.2_Lamini/2025-02-03_05-55-09_rNone_lr3e-05_lamini_r2'
    min_n = 500
    max_n = 6000
    step = 500
    remove_state_files(root_dir, min_n, max_n, step)
    copy_tokenizer(root_dir, min_n, max_n, step)

    min_n=5920
    remove_state_files(root_dir, min_n, max_n, step)
    copy_tokenizer(root_dir, min_n, max_n, step)