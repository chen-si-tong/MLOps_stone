import click
import torch
import torch.nn.functional as F
from torch import optim
from model import MNISTModel
import pickle
import matplotlib.pyplot as plt


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="Learning rate to use for training")
@click.option("--epochs", default=5, help="Number of epochs for training")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print("Learning rate: ", lr)

    # Load train data
    with open("data/processed/train_dataset.pkl", "rb") as file:
        train_set = pickle.load(file)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Model initialization
    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "MLOps_test/models/final_model/mnist_model.pth")

    # plt.plot(epoch_losses, label="Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Curve")
    # plt.legend()
    # plt.savefig("reports/figures/training_loss_curve.png")
    # plt.show()


@click.command()
@click.argument("model_checkpoint", type=str)
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print("Model checkpoint: ", model_checkpoint)

    # Load test data

    with open("data/processed/test_dataset.pkl", "rb") as file:
        test_set = pickle.load(file)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    # Load model
    model = MNISTModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Evaluation
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)"
    )


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
