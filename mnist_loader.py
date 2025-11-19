import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

from tensor import Tensor


def load_mnist(
    path: str = "data", train: bool = True, normalize: bool = True
) -> tuple[Tensor, Tensor]:
    """
    Load MNIST dataset from local files.

    Args:
        path: Directory containing MNIST files
        train: If True, load training data; otherwise load test data
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        Tuple of (images, labels) as Tensors
        Images shape: (num_samples, 784) or (num_samples, 784) normalized
        Labels shape: (num_samples,)
    """
    mndata = MNIST(path)
    if train:
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()
    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)
    if normalize:
        images_np = images_np / 255.0
    images_tensor = Tensor(images_np, requires_grad=False)
    labels_tensor = Tensor(labels_np, requires_grad=False)
    return images_tensor, labels_tensor


if __name__ == "__main__":
    images, labels = load_mnist("data", train=True, normalize=True)
    num_images = 15
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()
    for i in range(num_images):
        img = images.data[i].reshape(28, 28)
        label = labels.data[i]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
