import argparse
import json
import os

import torch

from model import ModelArgs, Transformer


def model_expoert(model, filepath, version=0, dtype=torch.float32):
    pass


def load_gpt_oss_model(model_path):
    config = ModelArgs()
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        params = json.load(f)
    config.dim = params["hidden_size"]
    config.n_layers = params["num_hidden_layers"]
    config.n_experts = params["num_experts"]
    config.experts_per_token = params["experts_per_token"]
    config.n_heads = params["num_attention_heads"]
    config.n_kv_heads = params["num_key_value_heads"]
    config.vocab_size = params["vocab_size"]
    config.hidden_dim = params["intermediate_size"]
    config.swiglu_limit = params["swiglu_limit"]
    config.head_dim = params["head_dim"]
    config.sliding_window = params["sliding_window"]
    config.context_length = params["initial_context_length"]
    config.rope_theta = params["rope_theta"]
    config.rope_scaling_factor = params["rope_scaling_factor"]
    config.rope_ntk_alpha = params["rope_ntk_alpha"]
    config.rope_ntk_beta = params["rope_ntk_beta"]

    model = Transformer(config)
    # model = model.from_checkpoint(model_path)
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
