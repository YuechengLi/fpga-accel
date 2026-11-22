#!/usr/bin/env python3
"""Quantize Qwen2.5-Omni-3B (BF16 + F32 token2wav) to the repo int16 scheme.

Source: models/llm/qwen2.5-omni-3b/raw/ (hash-verified download,
commit d0b9f56). Output: models/llm/qwen2.5-omni-3b/quantized/ — same
shard layout and tensor names, dtype I16, plus model.safetensors.index.json;
quant.json (per-tensor e / scale / error stats) goes to this directory
(tracked). Quantized output is gitignored (models/.gitignore:
llm/*/quantized/).

Scheme (models/llm/llm_common.py, same as models/vision/common.py):
per tensor e = floor(log2(32767/max|W|)) clamped [0,15],
W_q = round(W * 2**e). The token2wav DiT/BigVGAN tensors are F32 in
the checkpoint; they go through the same scheme (F32 -> int16). No
activation calibration exists for this model, so there is no r_in /
r_out and no RTL shift; every tensor (weight, bias, norm, embedding,
codec head, DiT/BigVGAN) is quantized on its own scale.
`quant.json` (per-tensor e / scale / error stats) is written to this
directory (tracked); the int16 shards + index go to `quantized/`
(gitignored).

Run from this directory:  python3 extract_omni.py
Streams tensor-by-tensor (2M-element chunks); never loads a full
checkpoint. ~11.2 GiB in, ~10.3 GiB int16 out (F32 token2wav shrinks
4 bytes -> 2 bytes per param).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_common as L

HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    L.quantize_model(os.path.join(HERE, 'raw'),
                     os.path.join(HERE, 'quantized'),
                     'Qwen2.5-Omni-3B')
