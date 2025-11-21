import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

from tensor import Tensor


def load_mnist(
    path: str = "data",
    train: bool = True,
    normalize: bool = True,
    one_hot: bool = False,
    batch_size: int = 1,
) -> list[tuple[Tensor, Tensor]]:
    """
    Load MNIST dataset from local files.

    Args:
        path: Directory containing MNIST files
        train: If True, load training data; otherwise load test data
        normalize: If True, normalize pixel values to [0, 1]
        one_hot: If True, return one-hot encoded labels
        batch_size: The number of samples per batch.

    Returns:
        List of (image, label) tuples, where each is a Tensor.
        Image Tensor shape: (batch_size, 784)
        Label Tensor shape: (batch_size,) or (batch_size, 10) if one_hot=True
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

    if one_hot:
        num_classes = labels_np.max() + 1
        labels_np = np.eye(num_classes)[labels_np]

    num_samples = images_np.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    data = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        image_batch = Tensor(images_np[start:end], requires_grad=False)
        label_batch = Tensor(labels_np[start:end], requires_grad=False)
        data.append((image_batch, label_batch))

    return data


if __name__ == "__main__":
    data = load_mnist("data", train=True, normalize=True, batch_size=15)
    images_batch, labels_batch = data[0]
    num_images = images_batch.shape[0]
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()
    for i in range(num_images):
        img = images_batch.data[i].reshape(28, 28)
        label = labels_batch.data[i]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
