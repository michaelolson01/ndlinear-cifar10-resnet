import matplotlib.pyplot as plt

def plot_accuracy_comparison(epochs, baseline_acc, ndlinear_acc, save_path="accuracy_comparison.png"):
    """
    Plots test accuracy comparison between baseline and NdLinear models.

    Args:
        epochs (list): List of epoch numbers.
        baseline_acc (list): Accuracy values for baseline model.
        ndlinear_acc (list): Accuracy values for NdLinear model.
        save_path (str): File path to save the output plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_acc, label='Baseline ResNet18', marker='o')
    plt.plot(epochs, ndlinear_acc, label='NdLinear ResNet18', marker='s')
    plt.title('Test Accuracy Comparison: Baseline vs. NdLinear on CIFAR-10')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
