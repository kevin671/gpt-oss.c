import struct
from typing import Dict, Tuple

import tiktoken


def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )
    return tokenizer


def export_tiktoken_encoding_to_bin(enc: tiktoken.Encoding, out_path: str) -> None:
    """
    Write a tokenizer.bin compatible with your C loader:
      uint32 max_token_length
      then for id = 0..max_id:
        float score   (here: -rank for mergeables; 0.0 for specials)
        uint32 length
        bytes payload (no trailing NUL)
    Little-endian, 4-byte sizes.
    """
    # 1) Collect id -> (bytes, score)
    id_to_bytes: Dict[int, bytes] = {}
    id_to_score: Dict[int, float] = {}

    # Mergeable ranks: mapping bytes -> rank (id)
    for b, rank in enc._mergeable_ranks.items():
        # rank is the token id for mergeable tokens
        tok_id = int(rank)
        id_to_bytes[tok_id] = b
        # Use negative rank as "score" to preserve L2R priority like llama exporter used scores
        id_to_score[tok_id] = float(-rank)

    # Special tokens: mapping string -> id
    for s, tok_id in enc._special_tokens.items():
        b = s.encode("utf-8")
        id_to_bytes[tok_id] = b
        # Specials don't participate in BPE merges; any float is fine. Use 0.0.
        id_to_score[tok_id] = 0.0

    max_id = max(id_to_bytes.keys())
    vocab_size = max_id + 1

    # Sanity: ensure no holes (or at least fill with empty tokens if needed)
    missing = [i for i in range(vocab_size) if i not in id_to_bytes]
    if missing:
        # If gaps exist, you can either raise or fill with empty bytes.
        # It's safer to raise so you know what's missing.
        raise ValueError(f"Tokenizer IDs are not contiguous. Missing IDs: {missing[:10]} ... total={len(missing)}")

    # 2) Compute max_token_length
    max_token_length = max(len(id_to_bytes[i]) for i in range(vocab_size))

    # 3) Write the binary (little-endian, fixed sizes)
    with open(out_path, "wb") as f:
        # uint32 max_token_length
        f.write(struct.pack("<I", max_token_length))
        for i in range(vocab_size):
            b = id_to_bytes[i]
            s = id_to_score[i]
            f.write(struct.pack("<fI", s, len(b)))
            f.write(b)


if __name__ == "__main__":
    enc = get_tokenizer()
    export_tiktoken_encoding_to_bin(enc, "tokenizer.bin")
    print("wrote tokenizer.bin")
