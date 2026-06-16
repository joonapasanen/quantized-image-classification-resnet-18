import argparse
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import v2

from train.model import model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def verify(weights: str, onnx_path: str, batch_size: int):
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    session = ort.InferenceSession(onnx_path)

    dataset = CIFAR10("data/", train=False, download=True, transform=test_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x, Y = next(iter(loader))

    with torch.no_grad():
        pt_out = model(x)

    ort_out = session.run(None, {"input": x.numpy()})[0]

    pt_preds = pt_out.argmax(dim=1).numpy()
    ort_preds = ort_out.argmax(axis=1)

    print("PyTorch preds:", pt_preds.tolist())
    print("ONNX preds:   ", ort_preds.tolist())
    print("labels:       ", Y.tolist())
    print("preds match:  ", (pt_preds == ort_preds).all())
    print("max abs diff: ", np.abs(pt_out.numpy() - ort_out).max())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="path to .pt file")
    parser.add_argument("--onnx", default="results/model.onnx", help="path to .onnx file")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    verify(args.weights, args.onnx, args.batch_size)
