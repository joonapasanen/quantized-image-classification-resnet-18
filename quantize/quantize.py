import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import v2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

calibration_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CifarCalibrationDataReader(CalibrationDataReader):
    def __init__(self, num_samples: int):
        dataset = CIFAR10("data/", train=True, download=True, transform=calibration_transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # onnx needs a representative dataset to measure activation value ranges (200 by default)
        self.batches = [{"input": x.numpy()} for x, _ in list(loader)[:num_samples]]
        self.iterator = iter(self.batches)

    def get_next(self):
        return next(self.iterator, None)

def quantize(input_path: str, output_path: str, num_samples: int):
    reader = CifarCalibrationDataReader(num_samples)

    quantize_static(
        input_path,
        output_path,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    print(f"quantized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/model.onnx")
    parser.add_argument("--output", default="results/model_quantized.onnx")
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()
    quantize(args.input, args.output, args.num_samples)
