<div align="center">

<a href='https://github.com/UCSC-REAL/FLAT'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://www.arxiv.org/pdf/2410.11143'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
# FLAT: LLM Unlearning via Loss Adjustment with Only Forget Data 

FLAT is a "flat" loss adjustment approach which addresses these issues by maximizing f-divergence between the available template answer and the forget answer only w.r.t. the forget data. The variational form of the defined f -divergence theoretically provides a way of loss adjustment by assigning different importance weights for the learning w.r.t. template responses and the forgetting of responses subject to unlearning. Empirical results demonstrate that our approach not only achieves superior unlearning performance compared to existing methods but also minimizes the impact on the model’s retained capabilities, ensuring high utility across diverse tasks, including copyrighted content unlearning on Harry Potter dataset and MUSE Benchmark, and entity unlearning on the TOFU dataset.

</div>


## 🎉🎉 News 
- [x] [2025.01] 👏👏 Accepted by **ICLR 2025**.
- [x] [2024.10] 🚀🚀 Release the paper of [**FLAT**](https://www.arxiv.org/pdf/2410.11143).

## Installation

```
conda create -n flat python=3.10
conda activate flat
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install natsort
pip install sacrebleu
pip install sentencepiece
```

## Task 1: Unlearning on Harry Potter

### Finetune the original model

```
# OPT-2.7b lr=1e-5
python finetune.py --model_name facebook/opt-2.7b

# finetune llama2-7b
python finetune.py --model_name meta-llama/Llama-2-7b-hf
```

After obtaining the finetuned model, we need to change the `hp_ft_model_path` in model_config.yaml. When unlearning, the code will load the finetuned model as the original model.

### Unlearn
```
master_port=18765
model=llama2-7b
lr=2e-7
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port unlearn.py --config-name=forget.yaml batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

### Metric & Evaluation

Unlearning efficacy: 

- BLEU on Harry Potter completion
- Rouge-L on Harry Potter completion

Utility:

- Perplexity on Wikitext
- Zero-shot Accuracy on benchmarks
- Zero-shot Accuracy on TruthfulQA

```
python evaluate.py --method_name $your_name --model_save_dir $your_model_path
```

## Task 2 & Task 3: Unlearning on TOFU and MUSE-NEWS
Please refer to [**TOFU**](https://github.com/locuslab/tofu) and [**MUSEBENCH**](https://github.com/swj0419/muse_bench).

## Citing FLAT

If you find our codebase and dataset beneficial, please cite our work:
```
@article{wang2024llm,
  title={LLM Unlearning via Loss Adjustment with Only Forget Data},
  author={Wang, Yaxuan and Wei, Jiaheng and Liu, Chris Yuhao and Pang, Jinlong and Liu, Quan and Shah, Ankit Parag and Bao, Yujia and Liu, Yang and Wei, Wei},
  journal={arXiv preprint arXiv:2410.11143},
  year={2024}
}
```

## Thanks
We thank the codebase from [**TOFU**](https://github.com/locuslab/tofu), [**SOUL**](https://github.com/OPTML-Group/SOUL), and [**MUSEBENCH**](https://github.com/swj0419/muse_bench).
