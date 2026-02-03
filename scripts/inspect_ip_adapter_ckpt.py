import argparse
from collections import defaultdict
from typing import Dict, Tuple, Set

import torch
from safetensors.torch import load_file


def split_modules(flat_state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    combined: Dict[str, Dict[str, torch.Tensor]] = {}
    for combo_key, value in flat_state.items():
        parts = combo_key.split(".")
        module = parts[0]
        key = ".".join(parts[1:])
        combined.setdefault(module, {})[key] = value
    return combined


def main():
    p = argparse.ArgumentParser(description="Inspect an IP-Adapter .safetensors checkpoint and suggest config values.")
    p.add_argument("ckpt", type=str, help="Path to IP-Adapter checkpoint (.safetensors)")
    args = p.parse_args()

    flat = load_file(args.ckpt, device="cpu")
    combined = split_modules(flat)

    modules = list(combined.keys())
    print("Modules:", modules)

    # Heuristics for ip vs ip+
    image_proj_keys = set(combined.get("image_proj", {}).keys())
    ip_adapter_keys = set(combined.get("ip_adapter", {}).keys())

    plus_markers = ("layers.", "attn", "to_q", "to_k", "to_v", "proj_in", "proj_out", "ff", "resampler")
    looks_plus = any(any(m in k for m in plus_markers) for k in image_proj_keys)
    looks_ip = any(k.startswith("proj.") for k in image_proj_keys)

    guessed_type = "ip+"
    if looks_ip and not looks_plus:
        guessed_type = "ip"
    elif looks_plus and not looks_ip:
        guessed_type = "ip+"
    elif looks_ip and looks_plus:
        guessed_type = "ip+"

    # Infer num_tokens when possible
    num_tokens_guess = None
    cross_dim_guess = None
    if "proj.weight" in image_proj_keys:
        w = combined["image_proj"]["proj.weight"]
        out_dim, in_dim = w.shape
        # out_dim ~= num_tokens * cross_attention_dim
        # try a few common cross dims
        for cross_dim in (3072, 4096, 2048, 1280, 1024, 768, 640):
            if out_dim % cross_dim == 0:
                nt = out_dim // cross_dim
                if 1 <= nt <= 64:
                    num_tokens_guess = nt
                    cross_dim_guess = cross_dim
                    break

    # Inspect ip_adapter to_k_ip shapes
    to_k_shapes: Set[Tuple[int, int]] = set()
    for k, v in combined.get("ip_adapter", {}).items():
        if k.endswith("to_k_ip.weight"):
            to_k_shapes.add(tuple(v.shape))

    print("\nimage_proj keys sample:", sorted(list(image_proj_keys))[:10])
    print("ip_adapter keys sample:", sorted(list(ip_adapter_keys))[:10])
    print("\nUnique to_k_ip.weight shapes:", sorted(list(to_k_shapes))[:10], ("(+more)" if len(to_k_shapes) > 10 else ""))

    # Flux hint: typically single uniform hidden_size 3072x3072
    flux_like = (len(to_k_shapes) == 1 and next(iter(to_k_shapes))[0] in (3072, 4096) and next(iter(to_k_shapes))[0] == next(iter(to_k_shapes))[1])

    print("\nGuesses:")
    print(f"  - adapter.type: {guessed_type}")
    if num_tokens_guess is not None:
        print(f"  - num_tokens: {num_tokens_guess} (from image_proj.proj.weight out_dim, cross_dim≈{cross_dim_guess})")
    else:
        # Common defaults
        print("  - num_tokens: (could not infer) common defaults: ip=4, ip+=16")
    print(f"  - looks_flux_like_weights: {flux_like}")

    print("\nSuggested YAML snippet:")
    print("adapter:")
    print(f"  type: \"{guessed_type}\"")
    print("  train: false")
    print(f"  name_or_path: \"{args.ckpt}\"")
    print("  image_encoder_arch: \"clip\"")
    print("  image_encoder_path: \"openai/clip-vit-large-patch14\"")
    print("  clip_layer: \"penultimate_hidden_states\"")
    if num_tokens_guess is not None:
        print(f"  num_tokens: {num_tokens_guess}")
    else:
        print(f"  num_tokens: {16 if guessed_type == 'ip+' else 4}")


if __name__ == "__main__":
    main()

