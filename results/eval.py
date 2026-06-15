import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import v2

from train.model import model

DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def evaluate(weights: str):
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = CIFAR10("data/", train=False, download=True, transform=test_transforms)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, Y in loader:
            x, Y = x.to(DEVICE), Y.to(DEVICE)
            X = model(x)
            total_loss += loss_fn(X, Y).item()
            correct += (X.argmax(dim=1) == Y).sum().item()
            total += Y.size(0)

    print(f"weights: {weights}")
    print(f"loss:       {total_loss / len(loader):.4f}")
    print(f"accuracy:   {correct / total * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="path to .pt file")
    args = parser.parse_args()
    evaluate(args.weights)
