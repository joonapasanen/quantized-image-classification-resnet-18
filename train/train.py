import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from pathlib import Path

from model import model

RESULTS_DIR = Path("results")


DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

model.to(DEVICE)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(32, padding=4),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = CIFAR10("data/", train=True, download=True, transform=train_transforms)
test_dataset = CIFAR10("data/", train=False, download=True, transform=test_transforms)

# hyperparams for training
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 15
WARMUP_EPOCHS = 2
BATCH_SIZE = 256
NUM_WORKERS = 4

def finetune_resnet():
    model.train()

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    optimizer = SGD(model.parameters(), LEARNING_RATE, MOMENTUM, weight_decay=WEIGHT_DECAY)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, EPOCHS - WARMUP_EPOCHS, LEARNING_RATE * 0.1)

    # use warmup first, then cosine
    scheduler = SequentialLR(optimizer, [warmup, cosine], [WARMUP_EPOCHS])

    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    # training loop
    for i in range(EPOCHS):

        # iterate through the data in batches
        train_loss = 0.0
        for x, Y in train_loader:
            model.zero_grad()

            x, Y = x.to(DEVICE), Y.to(DEVICE)

            X = model(x)
            loss = loss_fn(X, Y)

            loss.backward()

            if not torch.isfinite(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).all():
                model.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation loss
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, Y in test_loader:
                x, Y = x.to(DEVICE), Y.to(DEVICE)

                X = model(x)
                val_loss += loss_fn(X, Y).item()
                correct += (X.argmax(dim=1) == Y).sum().item()
                total += Y.size(0)

        val_loss /= len(test_loader)
        val_acc = correct / total

        model.train()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), RESULTS_DIR / "model_best.pt")
        
        print(f"EPOCH: {i}  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), RESULTS_DIR / "model_last.pt")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

if __name__ == "__main__":
    finetune_resnet()
