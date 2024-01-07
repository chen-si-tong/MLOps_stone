import torch
import torch.nn.functional as F
from torch import optim
from model import MNISTModel
import pickle
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
@hydra.main(config_path="/Users/chensitong/MLOps/project/MLOps_test/conf", config_name="main.yaml",version_base="1.3.2")
def main(config): #这里控制train还是evaluate
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams_model = config.model
    hparams_training = config.training

    model = MNISTModel(hparams_model)
    # train(hparams_training,model)
    evaluate(hparams_training,model)
    


def train(hparams_training,model):
    """Train a model on MNIST."""


    # Load train data
    with open(hparams_training.train_data_path, "rb") as file:
        train_set = pickle.load(file)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams_training.batch_size, shuffle=True)

    # Model initialization



    optimizer = optim.Adam(model.parameters(), lr=hparams_training.lr)

    # Training loop
    epoch_losses = []
    model.train()
    for epoch in range(hparams_training.epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "final_model/mnist_model.pth")

    plt.plot(epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("/Users/chensitong/MLOps/project/MLOps_test/reports/figures/training_loss_curve.png")
    plt.show()

def evaluate(hparams_training,model):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print("Model checkpoint: ", hparams_training.model_checkpoint)


    # Load test data

    with open(hparams_training.test_data_path, "rb") as file:
        test_set = pickle.load(file)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hparams_training.batch_size , shuffle=False)

    # Load model
    # model = MNISTModel()
    model.load_state_dict(torch.load(hparams_training.model_checkpoint))
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




if __name__ == "__main__":
    main()

