import numpy as np

from loss import cross_entropy_loss
from mnist_loader import load_mnist
from model import MLP
from optim import SGD
from tensor import Tensor


def main():
    input_dim = 784
    hidden_size = 128
    output_dim = 10
    num_hidden_layers = 1
    learning_rate = 0.01
    epochs = 5

    print("Loading MNIST data...")
    train_data = load_mnist("data", train=True, normalize=True, one_hot=True)
    test_data = load_mnist("data", train=False, normalize=True, one_hot=True)
    print("MNIST data loaded.")

    model = MLP(
        input_dim,
        output_dim,
        num_hidden=num_hidden_layers,
        hidden_size=hidden_size,
    )

    optimizer = SGD(model.parameters(), learning_rate)

    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")

    print("\nStarting training...")
    num_train_samples = len(train_data)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for image, label in train_data:
            predictions = model.forward(image)

            loss = cross_entropy_loss(predictions, label)
            epoch_loss += loss.data

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print(
            f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {epoch_loss / num_train_samples:.4f}"
        )
    print("Training finished.")

    print("\nStarting evaluation...")
    correct_predictions = 0
    num_test_samples = len(test_data)
    for image, label in test_data:
        predictions = model.forward(image)
        predicted_label = np.argmax(predictions.data, axis=1)
        true_label = np.argmax(label.data, axis=1)
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_samples
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
