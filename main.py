import copy

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
    epochs = 500
    batch_size = 32
    patience = 3

    print("Loading MNIST data...")
    train_data = load_mnist(
        "data", train=True, normalize=True, one_hot=True, batch_size=batch_size
    )
    test_data = load_mnist(
        "data", train=False, normalize=True, one_hot=True, batch_size=batch_size
    )
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
    print(f"Batch Size: {batch_size}")
    print(f"Patience: {patience}")

    print("\nStarting training...")
    num_train_batches = len(train_data)
    num_test_batches = len(test_data)

    best_test_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for image, label in train_data:
            predictions = model.forward(image)

            loss = cross_entropy_loss(predictions, label)
            epoch_loss += loss.data

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        test_loss = 0.0
        for image, label in test_data:
            predictions = model.forward(image)
            loss = cross_entropy_loss(predictions, label)
            test_loss += loss.data

        avg_train_loss = epoch_loss / num_train_batches
        avg_test_loss = test_loss / num_test_batches
        print(
            f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Test loss did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")

    if best_model_state:
        print("Loading best model for evaluation.")
        model.load_state_dict(best_model_state)

    print("\nStarting evaluation...")
    correct_predictions = 0
    test_data_acc = load_mnist(
        "data", train=False, normalize=True, one_hot=True, batch_size=1
    )
    num_test_samples = len(test_data_acc)
    for image, label in test_data_acc:
        predictions = model.forward(image)
        predicted_label = np.argmax(predictions.data, axis=1)
        true_label = np.argmax(label.data, axis=1)
        correct_predictions += np.sum(predicted_label == true_label)

    accuracy = correct_predictions / num_test_samples
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
