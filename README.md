# AnchorCoder

## Quick Start
### Environment Setup
```bash
conda env create -f environment.yml
conda activate anchorcoder
```

### To evaluate on HumanEval
```bash
cd scripts
bash humaneval.sh
```

### To evaluate on HumanEvalPlus
```bash
cd scripts
bash humanevalplus.sh
```

### To evaluate on MBPP
```bash
cd scripts
bash mbpp.sh
```

### Training from scratch
1. Choose your dataset
2. Add anchor for your dataset (ref add_anchor.py) 
3. Train your model by:
```bash
accelerate launch --multi_gpu --num_processes your_gpu_num --mixed_precision bf16 train_multigpu.py 
```

### Computational Overhead
* We pre-trained AnchorCoder using LoRA with a rank of 32 (total model dimension of 4096) for 1 epoch on the CodeSearchNet dataset and fine-tuned it for 2 epochs on the CodeHarmony dataset, totaling 150M tokens, which is significantly less than the computational requirements for pre-training. In addition, we have only optimized the parameters related to self-attention (W_Q, W_K, W_V, W_O), which reduces the computational load and mitigates the risk of data leakage.




