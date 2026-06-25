**Contribution of this work**

In this project, we perform a modular reproduction and a comparative analysis of the Cross-Domain Sequential Recommendation landscape. The core of our work is a from-scratch mathematical implementation of Pareto-aware learning framework. We manually expose and manipulate internal Transformer attention matrices to compute the cross-domain attention mass and solving the resulting constrained optimization problem via the Frank-Wolfe algorithm. This ensures a deep understanding of the gradient-balancing mechanism that standard black-box libraries cannot provide.
Complementing this algorithmic depth is a comprehensive native pipeline reconstruction and engineering effort. We refactored the data and training workflows to bridge the architectural gaps between different models, enabling C2DSR and SyNCRec to be trained and evaluated on the exact dataset configurations originally prepared for AutoCDSR, ensuring that our performance comparisons are conducted on a strictly identical data distribution.
Finally, to rigorously validate the efficacy of the Pareto-optimal approach, we conducted extensive baseline benchmarking against two dominant CDSR paradigms: the GNN-based collaborative and contrastive learning of C2DSR, and the MoE-based decoupling and negative transfer gap mitigation of SyNCRec. Through this systematic three-way comparison on the KuaiRand-1K dataset, we provide an evaluation of how different architectural philosophies navigate the trade-off between cross-domain knowledge sharing and potential domain interference.

**How to complie and excute this project.**

First install the requirements environment

`pip install -r requirements.txt`


Then run the script following, or you can simply run the code by yourself following the same order with the script.

```bash
#!/bin/bash

echo "Step 1: Remapping data..."
python remap.py

echo "Step 2: Training model..."
python train_autocdsr_full.py 

echo "Step 3: Testing model..."
python test_autocdsr_sampled.py 
```

**Source Code Description**
autocdsr_bert4rec.py: Model

dataloader.py: Dataloader to load training, evaluation and testing data

masking.py: Masking script to build bert4rec style training, evaluation and testing datasets.

remap.py: Remap data to prevent sparse matrix when using slicing data 

test_autocdsr_sampled.py: Test script for trained model

train_autocdsr_full.py: Training script for autocdsr model

source code/data: Dataset for this project

**Runing Platform**

Windows: Windows 11 with cuda 12+ and discrete GPU (RTX3060, RTX5070Ti and some other consumer-grade GPU)
Linux: HKUST SuperPOD

**Example Output**
You can find some example output of this project in `output.txt`
