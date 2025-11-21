import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

from tensor import Tensor


def load_mnist(
    path: str = "data",
    train: bool = True,
    normalize: bool = True,
    one_hot: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    """
    Load MNIST dataset from local files.

    Args:
        path: Directory containing MNIST files
        train: If True, load training data; otherwise load test data
        normalize: If True, normalize pixel values to [0, 1]
        one_hot: If True, return one-hot encoded labels

    Returns:
        List of (image, label) tuples, where each is a Tensor.
        Image Tensor shape: (1, 784)
        Label Tensor shape: (1,) or (1, 10) if one_hot=True
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

    data = []
    for i in range(images_np.shape[0]):
        image_tensor = Tensor(images_np[i : i + 1], requires_grad=False)
        label_tensor = Tensor(labels_np[i : i + 1], requires_grad=False)
        data.append((image_tensor, label_tensor))

    return data


if __name__ == "__main__":
    data = load_mnist("data", train=True, normalize=True)
    num_images = 15
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    axes = axes.flatten()
    for i in range(num_images):
        image_tensor, label_tensor = data[i]
        img = image_tensor.data.reshape(28, 28)
        label = label_tensor.data
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
