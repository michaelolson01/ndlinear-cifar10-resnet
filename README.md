# CIFAR-10 Classification with NdLinear-Enhanced ResNet

This project benchmarks the performance of a ResNet-18 model using `NdLinear` in place of the standard linear classification layer on the CIFAR-10 dataset.

## 🔍 Goal

Evaluate whether NdLinear improves generalization or representation when used in the final classification layer of a CNN.

## 🧠 Architecture

- Base model: `torchvision.models.resnet18(pretrained=False)`
- Modified: Final `nn.Linear` layer replaced with `NdLinear`
- Dataset: CIFAR-10 (10 classes of 32x32 color images)

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python train.py

## 📊 Results (To Be Filled In)

| Model             | Accuracy (%) | Loss |
| ----------------- | ------------ | ---- |
| Baseline ResNet18 | XX.X         | X.XX |
| NdLinear ResNet18 | XX.X         | X.XX |

## 📦 Requirements

- torch
- torchvision
- ndlinear (via pip install git+https://github.com/ensemble-core/NdLinear)
- matplotlib
- tqdm
