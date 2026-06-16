import argparse
import torch

from model import model

def export(checkpoint: str, output: str):
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    # match cifar10 dims
    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy_input,
        output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"exported to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to .pt file")
    parser.add_argument("--output", default="results/model.onnx", help="output .onnx path")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
