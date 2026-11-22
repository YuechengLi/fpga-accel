# Qwen2.5-Omni-3B — verified + registered (int16 quantized)

> **LICENSE: RESEARCH ONLY.** This model is released under the Qwen
> Research License Agreement for **non-commercial use only**. Commercial
> use requires a separate license from Alibaba's Qwen Team. Research use
> (this repo's purpose) is permitted.

- Source: https://huggingface.co/Qwen/Qwen2.5-Omni-3B
- Fetched: 2026-11-21 (PST), via `bash models/llm/fetch_llm.sh` (repo root)
- Raw dir: `models/llm/qwen2.5-omni-3b/raw/` — total 11.17 GiB (11,989,103,936 bytes)
- Status: **VERIFIED + REGISTERED 2026-11-22** — architecture checked
  against `config.json`, all 2544 tensors quantized to the repo int16
  scheme, single-layer forward checked. Raw weights are gitignored
  (`models/.gitignore`: `llm/*/raw/`); quantized weights are also
  gitignored (`llm/*/quantized/`). Only this README, `quant.json`,
  and the scripts are tracked.

## Architecture (verified from the checkpoint, not just config.json)

- Thinker (text): 36 layers (0–35 present), hidden 2048, SwiGLU
  intermediate 11008, 16 Q heads / 2 KV heads, head_dim 128, vocab
  151936, `tie_word_embeddings: false` — `thinker.lm_head`
  [151936, 2048] IS present. RoPE theta 1,000,000, mRoPE section
  [16, 24, 24]. Per-layer shapes (layer 0): q_proj [2048, 2048],
  k/v_proj [256, 2048], o_proj [2048, 2048], gate/up [11008, 2048],
  down [2048, 11008].
- Thinker vision encoder: 32 blocks, hidden 1280, intermediate
  3420, 16 heads, separate q/k/v projections (not fused QKV),
  patch 14, out_hidden 2048, spatial_merge 2, full-attention blocks
  at [7, 15, 23, 31].
- Thinker audio encoder (Whisper-based): 32 layers, d_model 1280,
  20 heads, ffn 5120, 128 mel bins, output dim 2048.
- Talker: 24 layers (0–23 present), hidden 896, intermediate 4864,
  14 Q heads / 2 KV heads, head_dim 64, vocab 8448,
  embedding_size 2048 (`talker.model.embed_tokens` [8448, 2048]),
  no `lm_head` (output goes through `talker.codec_head` instead),
  `thinker_to_talker_proj` bridges Thinker hidden states to Talker.
- token2wav: DiT depth 22, dim 1024, 16 heads + BigVGAN
  (upsample rates [5, 3, 2, 2, 2, 2], mel_dim 80). All 809
  token2wav tensors are **F32** in the checkpoint (the only F32
  tensors in this model; everything else is BF16).
- Total: **5,537,120,672 params** (5.54B), 2544 tensors: thinker
  1442 / 4,703,464,448, talker 293 / 384,604,928, token2wav 809 /
  449,051,296. The "3B" counts only the Thinker text backbone;
  Talker + encoders + token2wav make the checkpoint ~5.5B params
  (the earlier ~6B estimate was close; the measured total is 5.54B).

## Quantization (repo scheme, `models/llm/llm_common.py`)

Per tensor `e = floor(log2(32767 / max|W|))` clamped [0, 15],
`W_q = round(W * 2**e)`, int16, `float = W_q * 2**(-e)`. Same formula
as `models/vision/common.py` `quantize_layer`, with the LLM
differences: no activation calibration exists (no forward on real
data), so there is no `r_in` / `r_out` and no RTL shift — every
tensor (weight, bias, norm, embedding, codec head, DiT/BigVGAN) is
quantized on its own per-tensor scale, recorded in `quant.json`.
The F32 token2wav tensors go through the same scheme (F32 -> int16).
Biases are quantized as standalone tensors (not folded with an
activation scale).

- Output: `quantized/` — same shard layout and tensor names as raw,
  dtype I16, plus `model.safetensors.index.json`; 10.31 GiB int16
  (2544 tensors, 151 s; the F32 token2wav part shrinks 4 -> 2 bytes
  per param). `quant.json` (this dir, tracked) holds the per-tensor
  `e` / `scale` / `max_abs` / `max_err` / `rms_err` / `snr_db` /
  `clipped`.
- The [0, 15] clamp binds for most weight tensors (2044/2544 at
  e=15): `max|W|` is typically 0.2–0.7, so the unclamped e would
  exceed 15. At e=15 the LSB is 2^-15 and the max weight error is
  half an LSB (1.53e-05). Zero tensors clipped.
- Per-type error (SNR of dequantized vs float; median / min):
  thinker text q/k/v/o 75.9 / 70.6 dB, mlp 76.3 / 60.2 dB, norms
  121.9 / 97.7 dB, embed/lm_head 75.2 dB, vision 81.1 / 63.8 dB,
  audio 75.5 / 50.9 dB, talker layers 75.4 / 50.3 dB, talker
  embed/codec 80.4 / 61.4 dB, DiT 82.9 / 53.7 dB, BigVGAN 93.2 /
  54.6 dB. The low minima (~50 dB) are small tensors with a few
  large outliers (most values quantize finely; the outlier sets the
  scale) — the same heavy-tail effect seen in the YOLOv8n BN fold.

## Verification (`run_omni.py`, log `logs/llm/verify_omni.log`)

1. Inventory: 2544 tensors, names/shapes match raw, all dtype I16 — PASS.
2. Spot-check: 13 tensors of every type (thinker q/k/o, gate/down,
   norms; talker q/down; audio fc1; vision q; DiT ff; BigVGAN
   conv) — dequant max|err| reproduces `quant.json`, worst
   3.05e-05 — PASS. The two huge tensors (`thinker.lm_head`,
   `thinker.model.embed_tokens`, 311M params each) checked chunked
   (first 2M elements), max|err| 1.53e-05 — PASS.
3. Single-layer forward: thinker text layer 0, 4 fixed token ids
   [151644, 872, 1234, 4567], float (raw) vs dequantized int16 —
   max|err| 1.63e-03, rms_err 2.29e-04, relative RMS 5.34e-04
   (output RMS 0.429) — PASS. RoPE in this check is standard RoPE
   (theta 1e6), not the model's mRoPE; the identical transform is
   applied in both passes, so the number measures
   weight-quantization error only. No full-model forward, no
   benchmark, no accuracy claim.

## Files

- `raw/` (gitignored): the three safetensors shards + configs (see
  manifest below). Note `spk_dict.pt` (259,544 bytes) is a PyTorch
  speaker-dictionary file, NOT loaded in this verification.
- `quantized/` (gitignored): int16 shards + index.
- `quant.json` (tracked): per-tensor scales + error stats.
- `extract_omni.py`: quantize raw -> quantized (streaming, ~2.5 min).
- `run_omni.py`: inventory + spot-check + layer-0 forward check.
- `README.md`: this file.

## Limits / open items

- Weight-only quantization; no activation calibration, no
  per-channel scales. End-to-end quality (speech/text) is NOT
  measured — that needs a real eval harness, not done.
- The Talker + token2wav path (speech synthesis) is quantized but
  never run — the layer-0 check covers the thinker text decoder
  only. Audio-encoder and vision-encoder layers are quantized and
  spot-checked, not forward-checked.
- The compiler graph capture for this model (Thinker-Talker split,
  cross-modal routing) is not done; the quantized weights are the
  input to that work, not the result.
- Dual-card TP=2 decode targets (~300 tok/s roofline) were computed
  for the Thinker backbone class, not re-derived for this full
  checkpoint.

## File manifest (SHA256)

| file | bytes | SHA256 |
| --- | --- | --- |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| `LICENSE` | 7,387 | `ae8ef9d3fb476f735d9b9abeaab73b7f778b121c77427cd918ec3da01eefbc44` |
| `README.md` | 36,788 | `9b436a82898fa8f1e8559d1aea114052e7d471c4a5319676e033cbacc08ebcc1` |
| `added_tokens.json` | 579 | `e81a2cc3bd867a1217019eed202d1d8a07e1063ece716f22060dda14f6cc07d8` |
| `chat_template.json` | 1,313 | `1933f7ca08e6503460e981d9037ddc0b851196abfa3aceea52d6ee0c0fdfd9f0` |
| `config.json` | 13,183 | `20790f362c37a1718a3e764f597ae33dfc20177399762018dffaabf8321d4dd1` |
| `generation_config.json` | 74 | `b9acf52072249e0e1850541d51ca4c85a30b9d7e23c2f2093ec0fb4139059ac7` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `model-00001-of-00003.safetensors` | 4,994,850,880 | `349972cebff443030e9e96e1e930d7230979c6961cf046fa29fc580e4bcf49a6` |
| `model-00002-of-00003.safetensors` | 4,999,787,448 | `b29a76dfefb3aa33a5bd0faa19c681d039a44de57223653ea0e933df728c6d8c` |
| `model-00003-of-00003.safetensors` | 1,978,024,880 | `8f2138170a87ec604b53232491a96d6dcf19e0724aca37e69aace143e508b170` |
| `model.safetensors.index.json` | 241,817 | `5b7629198e2ef80e37612a491d9bfd71639d2f212632d36d8ab086922e74e129` |
| `preprocessor_config.json` | 667 | `b47055ce61463ce143e9aab741d55c0aa520801a0a5d63be73c5b17cecb6bc69` |
| `special_tokens_map.json` | 832 | `dc241604d085780a33f760f69c47e6622451104c20eee3cd3a12b6e884b43f44` |
| `spk_dict.pt` | 259,544 | `6a05609b28f5d42b7b748f0f07592545c8f1f6885b9ae8fff64baf56e86b2a18` |
| `tokenizer.json` | 11,421,870 | `8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99` |
| `tokenizer_config.json` | 6,469 | `569aa7a9171e36dfff80f0a7550ab0c9c09e46ac840fea5030641417394fb0d2` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
