# CIFAR-10 Classification with NdLinear

This project benchmarks the performance of a ResNet-18 model using [`NdLinear`](https://github.com/ensemble-core/NdLinear) in place of the standard linear classification layer on the CIFAR-10 dataset.

## 🔍 Objective

Evaluate how NdLinear compares to a standard `nn.Linear` layer in:
- Accuracy
- Validation loss
- Training time per epoch

## 📦 Requirements

You can install the required libraries with:

```bash
pip install torch torchvision tqdm matplotlib plotext
pip install git+https://github.com/ensemble-core/NdLinear.git
```

## 🚀 How to Run

```bash
cd src
python train.py
```

This will:
- Train both a baseline ResNet18 and a NdLinear-augmented version
- Log performance metrics
- Save PNG plots to disk
- Display terminal plots inline using `plotext`

## 📊 Results

Three visual comparisons are generated:
- `accuracy_comparison.png`
- `loss_comparison.png`
- `timing_comparison.png`

Each plot shows a comparison between the baseline and NdLinear-enhanced ResNet-18 models across all epochs.

## 📁 Project Structure

```
ndlinear-cifar10-resnet/
├── README.md
├── requirements.txt (optional)
└── src/
    ├── train.py
    └── utils.py
```

## ✍️ Notes

NdLinear maintains the multidimensional structure of input tensors better than standard flattening operations. This project evaluates whether that translates to better learning performance or efficiency on a moderately sized benchmark like CIFAR-10.
