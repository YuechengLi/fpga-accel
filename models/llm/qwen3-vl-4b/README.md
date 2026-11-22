# Qwen3-VL-4B-Instruct — verified + registered (int16 quantized)

- Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- License: Apache 2.0
- Fetched: 2026-11-21 (PST), via `bash models/llm/fetch_llm.sh` (repo root)
- Raw dir: `models/llm/qwen3-vl-4b/raw/` — total 8.28 GiB (8,887,292,732 bytes)
- Status: **VERIFIED + REGISTERED 2026-11-22** — architecture checked
  against `config.json`, all 713 tensors quantized to the repo int16
  scheme, single-layer forward checked. Raw weights are gitignored
  (`models/.gitignore`: `llm/*/raw/`); quantized weights are also
  gitignored (`llm/*/quantized/`). Only this README, `quant.json`,
  and the scripts are tracked.

## Architecture (verified from the checkpoint, not just config.json)

- Text decoder: 36 layers (0–35 present), hidden 2560, SwiGLU
  intermediate 9728, 32 Q heads / 8 KV heads, head_dim 128,
  vocab 151936, `tie_word_embeddings: true` (no separate `lm_head`
  tensor — the embedding is reused), RoPE theta 5,000,000, mRoPE
  section [24, 20, 20], 256k context.
- ViT: 24 blocks, hidden 1024, intermediate 4096, 16 heads, fused
  QKV (`attn.qkv`, [3072, 1024]), patch embed [1024, 3, 2, 16, 16]
  (temporal patch 2, 16x16 spatial), DeepStack visual indexes
  [5, 11, 17] (ViT layers whose outputs are injected), merger
  `linear_fc1` [4096, 4096] -> `linear_fc2` [2560, 4096].
- Total: **4,437,815,808 params** (4.44B), all BF16, 713 tensors:
  language_model 398 tensors / 4,022,468,096 params, visual 315
  tensors / 415,347,712 params.
- Per-layer shapes (layer 0, representative): q_proj [4096, 2560],
  k/v_proj [1024, 2560], o_proj [2560, 4096], gate/up_proj
  [9728, 2560], down_proj [2560, 9728]. LM layers carry no biases;
  ViT blocks do (qkv/proj/mlp/norm biases).

## Quantization (repo scheme, `models/llm/llm_common.py`)

Per tensor `e = floor(log2(32767 / max|W|))` clamped [0, 15],
`W_q = round(W * 2**e)`, int16, `float = W_q * 2**(-e)`. Same formula
as `models/vision/common.py` `quantize_layer`, with the LLM
differences: no activation calibration exists (no forward on real
data), so there is no `r_in` / `r_out` and no RTL shift — every
tensor (weight, bias, norm, embedding) is quantized on its own
per-tensor scale, recorded in `quant.json`. Biases are quantized as
standalone tensors (not folded with an activation scale).

- Output: `quantized/` — same shard layout and tensor names as raw,
  dtype I16, plus `model.safetensors.index.json`; 8.27 GiB int16
  (713 tensors, 112 s). `quant.json` (this dir, tracked) holds the
  per-tensor `e` / `scale` / `max_abs` / `max_err` / `rms_err` /
  `snr_db` / `clipped`.
- The [0, 15] clamp binds for most weight tensors (378/713 at e=15):
  `max|W|` is typically 0.2–0.7, so the unclamped e would exceed 15.
  At e=15 the LSB is 2^-15 and the max weight error is half an LSB
  (1.53e-05). Zero tensors clipped.
- Per-type error (SNR of dequantized vs float; median / min):
  LM q_proj 75.0 / 74.3 dB, k/v 75.1 / 72.1 dB, o 75.0 / 74.0 dB,
  gate/up/down 75.5 / 65.9 dB, LM norms 124.8 / 93.4 dB, embed
  74.8 dB, ViT qkv/proj 74.0 / 68.9 dB, ViT mlp/merger 75.1 /
  58.5 dB, ViT norms/bias 91.0 / 69.8 dB.

## Verification (`run_qwen3vl.py`, log `logs/llm/verify_qwen3vl.log`)

1. Inventory: 713 tensors, names/shapes match raw, all dtype I16 — PASS.
2. Spot-check: 12 tensors of every type (LM q/k/o, gate/down,
   norms, ViT qkv/mlp/merger) — dequant max|err| reproduces
   `quant.json`, worst 2.44e-04 — PASS. Embedding (389M params)
   checked chunked (first 2M elements), max|err| 1.53e-05 — PASS.
3. Single-layer forward: LM layer 0, 4 fixed token ids
   [151643, 872, 1234, 4567], float (raw) vs dequantized int16 —
   max|err| 4.42e-04, rms_err 5.68e-05, relative RMS 1.93e-04
   (output RMS 0.294) — PASS. RoPE in this check is standard RoPE
   (theta 5e6), not the model's interleaved mRoPE; the identical
   transform is applied in both passes, so the number measures
   weight-quantization error only. No full-model forward, no
   benchmark, no accuracy claim.

## Files

- `raw/` (gitignored): the two safetensors shards + configs (see
  manifest below).
- `quantized/` (gitignored): int16 shards + index.
- `quant.json` (tracked): per-tensor scales + error stats.
- `extract_qwen3vl.py`: quantize raw -> quantized (streaming, ~2 min).
- `run_qwen3vl.py`: inventory + spot-check + layer-0 forward check.
- `README.md`: this file.

## Limits / open items

- Weight-only quantization; no activation calibration, no
  per-channel scales. End-to-end quality (caption/VQA) is NOT
  measured — that needs a real eval harness, not done.
- The compiler graph capture for this model (attention, DeepStack
  injection, ViT merger) is not done; the quantized weights are the
  input to that work, not the result.
- Decode roofline for the VCU128 track (single board, 7B INT8
  65.7 tok/s class) was computed for generic 7B/13B models, not
  re-derived for this 4.4B multimodal model.

## File manifest (SHA256)

| file | bytes | SHA256 |
| --- | --- | --- |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| `README.md` | 7,133 | `a884e5e78f7d6f7bfe237f909dbc41a126542e259dc79d8ab33cc8980580ff79` |
| `chat_template.json` | 5,502 | `6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4` |
| `config.json` | 1,505 | `edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d` |
| `generation_config.json` | 269 | `8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44` |
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `model-00001-of-00002.safetensors` | 4,967,229,296 | `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9` |
| `model-00002-of-00002.safetensors` | 3,908,490,048 | `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` |
| `model.safetensors.index.json` | 64,742 | `58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer.json` | 7,032,403 | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `tokenizer_config.json` | 10,868 | `c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399` |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
