import matplotlib.pyplot as plt
import plotext as tplt

def plot_png_file(epochs, baseline, ndlinear, title="unknown", ylabel="unknown", save_path="plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline, label='Baseline ResNet18', marker='o')
    plt.plot(epochs, ndlinear, label='NdLinear ResNet18', marker='x')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[matplotlib] Plot saved to {save_path}")

def plot_terminal_preview(epochs, baseline, ndlinear, title="unknown", ylabel="unknown"):
    tplt.clear_figure()
    tplt.plot(epochs, baseline, label="Baseline")
    tplt.plot(epochs, ndlinear, label="NdLinear")
    tplt.title(title)
    tplt.xlabel("Epoch")
    tplt.ylabel(ylabel)
    tplt.show()

def plot_both(epochs, baseline, ndlinear, title, ylabel, save_path):
    plot_png_file(epochs, baseline, ndlinear, title, ylabel, save_path)
    plot_terminal_preview(epochs, baseline, ndlinear, title, ylabel)

def plot_both_accuracy(epochs, baseline_acc, ndlinear_acc, save_path="output/accuracy_comparison.png"):
    plot_both(epochs, baseline_acc, ndlinear_acc, title="Test Accuracy Comparison", ylabel="Accuracy (%)", save_path=save_path)

def plot_both_timing(epochs, baseline_time, ndlinear_time, save_path="output/timing_comparison.png"):
    plot_both(epochs, baseline_time, ndlinear_time, title="Training Time per Epoch", ylabel="Seconds", save_path=save_path)

def plot_both_loss(epochs, baseline_loss, ndlinear_loss, save_path="output/loss_comparison.png"):
    plot_both(epochs, baseline_loss, ndlinear_loss, title="Validation Loss per Epoch", ylabel="Loss", save_path=save_path)
