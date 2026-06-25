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