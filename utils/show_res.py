import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import re
import pandas as pd

def find_results_json(root_dir):
    results_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("results") and filename.endswith(".json"):
                results_files.append(os.path.join(dirpath, filename))
    return results_files

def calculate_average_from_json(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)

    output = {}
    for dataset, metrics in results.items():
        alias = metrics["alias"]  # 获取 alias 作为列名
        value = metrics.get("pct_stereotype,none", metrics.get("acc_norm,none", metrics.get("acc,none")))
        output[alias] = value

    values = list(output.values())
    avg_value = sum(values) / len(values) if values else 0  # 处理空值情况
    output['avg'] = avg_value
    return avg_value, output
def get_steps_num(path):
    match = re.search(r'checkpoint-(\d+)', path)

    if match:
        checkpoint_number = match.group(1)  # 提取到的数字
        return checkpoint_number
    else:
        return None



def plt_main(root_dir):
    import matplotlib.pyplot as plt
    results_files = find_results_json(root_dir)
    avg_values = []
    labels = []

    for file_path in results_files:
        avg_value = calculate_average_from_json(file_path)
        avg_values.append(avg_value)
        # 取文件路径中的数字部分作为标签
        label = os.path.basename(os.path.dirname(file_path))  # 获取文件夹名
        labels.append(label)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_values)
    plt.xlabel('step')
    plt.ylabel('Average Value')
    plt.title('Average Values from results.json')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def excel_main(root_dir):
    results_files = find_results_json(root_dir)
    res = {}
    for file_path in results_files:
        steps = get_steps_num(file_path)
        if steps is None:
            steps = file_path
        with open(file_path, 'r') as f:
            results = json.load(f)
        results = results["results"]
        output = {}
        for dataset, metrics in results.items():
            alias = metrics["alias"]  # 获取 alias 作为列名
            value = metrics.get("pct_stereotype,none",metrics.get("acc_norm,none", metrics.get("acc,none"))) 
            output[alias] = value
        values = list(output.values())
        avg_value = sum(values) / len(values) if values else 0  # 处理空值情况
        output['avg'] = avg_value
        res[steps] = output
    res = dict(sorted(res.items()))
    df = pd.DataFrame.from_dict(res, orient='index')
    print(df)
    print("max avg:", df["avg"].max(), "step:", df["avg"].idxmax() )
    output_file = os.path.join(root_dir,'evaluation_metrics.xlsx')
    df.to_excel(output_file, sheet_name='Metrics')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="这是一个示例程序，用于演示如何传递命令行参数。")
    parser.add_argument("--path", type=str, required=True, help="t")

    args = parser.parse_args()
    # print("r3")
    excel_main(args.path)
    # print("r2")
    # excel_main("out/packed_pretrain-llama3.2_1b/2025-01-21_04-25-07_rNone_lr5e-05_smollm-corpus_r2")

    # excel_main("/home/scccse/zhuhr/code/llm_prune/out/results/out__packed_pretrain-llama3.2_1b__2024-12-28_12-33-56_rNone_lr3e-05_smollm-corpus_r3__checkpoint-78500")