# GRAM (Generalized Region-Aware Mixture-of-Experts)

This repository contains the implementation of **GRAM**, a test-time adaptation framework for robust slum segmentation using satellite imagery. GRAM enables scalable, label-efficient mapping of informal settlements by combining region-specific experts and prediction consistency filtering.

## Prerequisites

1. Make sure you have the following dependencies installed:
```
- python==3.9
- pyTorch==1.7.1
- torchVision==0.8.2
- mmcv==1.2.7
- timm==0.3.2
- kornia==0.5.11
- openCV==4.5.1.48
- pyYAML==5.4.1
- numPy==1.20.3
- pandas==2.0.3
- sciPy==1.7.1
```


2. Download or prepare datasets and place them in the appropriate directory.

---

## Phase 1: Source Training

### (1) Train the MoE model

```bash
python main_moe.py \
    --train_meta ./metadata/train_metadata.csv \
    --test_meta ./metadata/UGA_test_metadata.csv \
    --epoch 10
```

### (2) Train the country classifier
This classifier is used to estimate the test country index:

```bash
python main_external_classifier.py \
    --train_meta ./metadata/train_metadata.csv
```

## Phase 2: Adaptation and Evaluation
### (1) Estimate target country index
Use the trained classifier to predict the region/domain of the test set:
```bash
python main_external_classifier_eval.py \
    --test_meta ./metadata/test_metadata.csv
```

### (2) Perform test-time adaptation
Adapt the source-trained MoE model to the target region using prediction consistency filtering:

```bash
python main_moe_pl_v3.py \
    --test_meta ./metadata/UGA_test_metadata.csv
```

## Notes

- The repository includes several ablation experiments for test-time adaptation and evaluation.
- Checkpoint files are stored in the `checkpoint/` directory.
- Metadata such as domain splits or city mappings can be found in `metadata/`.

