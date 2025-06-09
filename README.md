

This repository is the official implementation of "SDMPrune: Self-Distillation MLP Pruning for Efficient Large Language Models".

[[Paper]()]    [[BibTex](#Citation)]    [[Blog]()]    [[HuggingFace](https://huggingface.co/visresearch/SDMPrune/tree/main)]

## 1. Introduction

The gradient computation with one-hot labels ignore the potential predictions on other words, thus missing key information for generative capability of the original model. To address this issue, we introduce a self-distillation loss during the pruning phase (rather than post-training) to fully exploit the predictions of the original model, thereby obtaining more accurate gradient information for pruning. Moreover, we find that, compared to attention modules, the predictions of LLM are less sensitive to multilayer perceptron (MLP) modules, which take up more than 5× parameters (LLaMA3.2-1.2B). To this end, we focus on the pruning of MLP modules, to significantly compress LLM without obvious performance degradation. Experimental results on extensive zero-shot benchmarks demonstrate that our method significantly outperforms existing pruning methods.

![](./images/motivation.png)



## 2. Quick Start

### 2.1 Installation
```
conda create -n SDMPrune python=3.9
pip install -r requirement.txt
```

### 2.2 Prune LLMs

```
bash scripts/prune.sh
```
This script would compress the LLaMA3.2-1B model. You need to download [LLaMA3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) pretrained weights. The dataset would be automatically downloaded and sampled.



### 2.3 Train LLMs

```
bash scripts/run.sh
```
This script would compress the LLaMA3.2-1B model. You need to download [LLaMA3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) pretrained weights. The dataset would be automatically downloaded and sampled.



### 2.4 Evaluate results

```
bash scripts/eval_full.sh ckpt_path
bash scripts/eval_lora.sh ckpt_path adaptor_path
```



## 3. Main Results

+ **Zero-shot performance and Perplexity**

| Ratio | Method         |   PPL↓    |   ARCe    |   ARCc    |   BOOLQ   |   Crows   |   OBQA   |   PIQA    |   Race    |   SIQA    |    TQA    |   Wino    | Average↑  |
| :---: | :------------- | :-------: | :-------: | :-------: | :-------: | :-------: | :------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|  0%   | LLaMA3.2-1.2B  |   12.98   |   37.12   |   60.69   |   64.04   |   62.55   |   37.6   |   74.16   |   37.61   |   42.89   |   37.70   |   60.38   |   51.47   |
|       | Magnitude [17] |   36.01   |   25.35   |   47.33   |   58.07   |   56.93   |   30.4   |   66.33   |   32.03   |   41.21   |   40.12   |   53.46   |   45.12   |
|       | Wanda [39]     |   33.27   |   28.33   |   50.23   |   62.95   |   54.98   |   28.9   |   65.89   |   33.84   |   41.00   | **42.32** |   54.23   |   46.27   |
|  20%  | LLMPruner [23] |   28.92   |   29.61   |   51.00   |   62.07   |   57.54   | **33.6** |   67.19   |   33.68   |   40.89   |   41.68   |   56.59   |   47.39   |
|       | Compresso [16] |     -     |   27.05   |   48.53   |   59.14   | **58.56** |   27.4   |   66.97   |   33.21   |   40.23   |   43.74   |   55.64   |   46.05   |
|       | LoRAPrune [47] | **26.68** |   29.60   |   50.99   |   66.18   |   54.38   |   29.6   |   66.36   |   34.05   |   42.02   |   41.66   | **56.91** |   47.17   |
|       | SDMPrune(ours) |   26.96   | **31.14** | **55.22** | **67.58** |   57.84   |   32.4   | **70.29** | **35.41** | **42.73** |   40.20   |   56.59   | **48.94** |
|       | Magnitude [17] |   47.13   |   24.99   |   44.83   |   58.65   |   55.58   |   26.8   |   62.79   |   31.25   |   38.24   |   39.27   |   50.32   |   43.27   |
|       | Wanda [39]     |   45.73   |   25.35   |   45.62   |   59.02   |   52.32   |   27.3   |   63.19   |   31.73   |   37.73   |   43.12   |   54.65   |   44.01   |
|  30%  | LLMPruner [23] |   42.46   |   27.30   |   46.34   |   60.18   |   54.32   |   27.4   |   63.98   |   32.87   |   38.43   | **42.48** |   54.64   |   44.79   |
|       | Compresso [16] |     -     |   25.73   |   45.33   |   61.75   |   55.82   |   26.5   |   62.29   |   31.83   |   37.13   |   40.73   |   53.45   |   44.06   |
|       | LoRAPrune [47] |   40.11   |   28.07   |   46.38   |   59.61   |   56.29   |   27.6   |   64.92   |   32.37   |   37.32   |   41.59   |   54.38   |   44.86   |
|       | SDMPrune(ours) | **39.70** | **28.50** | **47.47** | **64.68** | **56.89** | **29.0** | **66.32** | **33.21** | **40.84** |   42.35   | **54.70** | **46.40** |
|       | Magnitude [17] |  100.66   |   24.18   |   40.37   |   56.12   |   51.42   |   25.0   |   61.07   |   31.24   |   37.25   |   43.01   |   49.81   |   41.95   |
|       | Wanda [39]     |   90.03   |   23.54   |   39.12   |   55.13   | **55.68** |   24.2   |   60.12   |   30.79   |   37.92   |   44.13   |   50.14   |   42.08   |
|  40%  | LLMPruner [23] |   76.7    |   25.09   |   39.94   |   58.47   |   52.06   | **28.0** |   60.45   |   30.33   |   38.69   | **44.90** |   51.14   |   42.91   |
|       | Compresso [16] |     -     |   24.35   |   40.13   |   59.01   |   54.09   |   26.1   |   61.92   |   30.61   |   37.64   |   42.53   |   51.09   |   42.75   |
|       | LoRAPrune [47] | **68.63** |   24.57   | **44.36** |   60.73   |   54.32   |   24.4   |   60.50   |   28.52   |   37.87   |   41.38   |   52.33   |   42.90   |
|       | SDMPrune(ours) |   70.12   | **26.02** |   42.63   | **65.38** |   52.59   |   25.6   | **63.44** | **32.25** | **38.74** |   43.30   | **52.17** | **44.21** |

+ **Comparison to other small-scale LLMs**

  | Model Name                     | #Params  | ARC-e    | ARC-c    | BOOLQ    | Crows    | OBQA     | PIQA     | Race     | SIQA     | TFIQA    | Wino     | Average  |
  | ------------------------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
  | ShearedLLaMA1.3B [44]          | 1.3B     | 29.1     | 54.4     | 62.0     | 63.7     | 34.4     | 73.4     | 36.3     | 41.3     | 36.8     | 58.1     | 49.0     |
  | TinyLLaMA1.1B [48]             | 1.1B     | 30.1     | 55.3     | 57.8     | 62.3     | 36.0     | 73.3     | 36.5     | 40.6     | 37.6     | 59.1     | 48.9     |
  | Pythia1.0B [4]                 | 1.1B     | 26.9     | 49.0     | 60.9     | 60.2     | 31.4     | 69.3     | 32.8     | 39.8     | **40.5** | 53.6     | 46.4     |
  | Falcon1.3B [1]                 | 1.3B     | 31.5     | 57.5     | 61.5     | 61.8     | 35.8     | 74.6     | 36.2     | 41.1     | 35.8     | 61.2     | 49.7     |
  | MobileLLM1.0B [22]             | 1.0B     | 33.5     | 58.5     | 65.6     | 60.4     | **36.6** | 73.6     | 34.6     | 41.3     | 38.3     | **63.3** | 50.6     |
  | Openelm1.1B [24]               | 1.1B     | 32.3     | 55.4     | 63.6     | 63.6     | 36.2     | **75.6** | 36.5     | 42.8     | 37.0     | 61.7     | 50.5     |
  | Opt1.3B [49]                   | 1.3B     | 27.8     | 51.2     | 57.2     | **65.8** | 32.6     | 70.9     | 34.2     | 40.4     | 38.7     | 59.4     | 47.8     |
  | MobiLLaMA1.2B [41]             | 1.2B     | 31.8     | 56.5     | 60.3     | 64.1     | 34.8     | 74.8     | 34.9     | 42.0     | 35.2     | 59.3     | 49.4     |
  | **SDMPrune (ours, ratio=20%)** | **1.0B** | **35.0** | **59.3** | **72.7** | 60.5     | 34.6     | 72.4     | **37.0** | **44.2** | 39.7     | 58.5     | **51.4** |



## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{zhu2025sdmprune,
  author  = {Zhu, Hourun and Shen, Chengchao},
  title   = {SDMPrune: Self-Distillation MLP Pruning for Efficient Large Language Models},
  year    = {2025},
}
```

