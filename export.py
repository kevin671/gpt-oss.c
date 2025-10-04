import argparse
import json
import os

import torch

from model import Transformer


def model_expoert(model, filepath, version=0, dtype=torch.float32):
    pass


def load_gpt_oss_model(model_path):
    model = Transformer.from_checkpoint(model_path, device="cpu")

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", type=str, help="the path to the gpt-oss model")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = load_gpt_oss_model(args.model_path)
    model_expoert(model, args.filepath, args.version, dtype=dtype)
