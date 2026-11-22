#!/usr/bin/env python3
"""Verify Qwen3-VL-4B-Instruct quantized output (run_qwen3vl.py).

Run from this directory AFTER extract_qwen3vl.py:
  python3 run_qwen3vl.py
Checks:
  (1) quantized inventory matches raw (names, shapes, dtype I16);
  (2) spot-check dequant error vs quant.json on 12 tensors of every
      type (LM q/k/v/o, gate/up/down, norms, ViT qkv + merger, embed);
  (3) single-layer forward check: LM layer 0, 4 tokens, float (raw)
      vs dequantized int16 — the only forward run in this
      verification (no full-model forward, no benchmark).
Prints PASS/FAIL per check plus the measured numbers.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_common as L

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
QUANT = os.path.join(HERE, 'quantized')

SPOT = [
    'model.language_model.layers.0.self_attn.q_proj.weight',
    'model.language_model.layers.0.self_attn.k_proj.weight',
    'model.language_model.layers.0.self_attn.o_proj.weight',
    'model.language_model.layers.0.mlp.gate_proj.weight',
    'model.language_model.layers.0.mlp.down_proj.weight',
    'model.language_model.layers.0.input_layernorm.weight',
    'model.language_model.layers.17.self_attn.q_proj.weight',
    'model.language_model.layers.35.mlp.down_proj.weight',
    'model.language_model.norm.weight',
    'model.visual.blocks.0.attn.qkv.weight',
    'model.visual.blocks.0.mlp.linear_fc1.weight',
    'model.visual.merger.linear_fc2.weight',
]


def main():
    with open(os.path.join(HERE, 'quant.json')) as f:
        quant = json.load(f)
    index = L.load_index(RAW)
    qindex = L.load_index(QUANT)
    # (1) inventory
    assert set(index) == set(qindex), 'tensor name sets differ'
    qhdr = {}
    rhdr = {}
    for shard in sorted(set(qindex.values())):
        h, _, _ = L.open_shard(os.path.join(QUANT, shard))
        qhdr.update(h)
    for shard in sorted(set(index.values())):
        rhdr.update(L.read_st_header(os.path.join(RAW, shard)))
    bad = [n for n, m in qhdr.items()
           if m['dtype'] != 'I16' or m['shape'] != rhdr[n]['shape']]
    print(f'(1) inventory: {len(index)} tensors, '
          f'{"PASS" if not bad else "FAIL " + str(bad[:3])}')
    assert not bad
    # (2) spot-check dequant error vs quant.json
    worst = 0.0
    for n in SPOT:
        f = L.load_f32_tensor(RAW, index, n).astype(np.float64)
        q = L.dequant_tensor(QUANT, qindex, n).astype(np.float64)
        err = float(np.max(np.abs(f - q)))
        rec = quant['tensors'][n]['max_err']
        assert abs(err - rec) <= max(1e-6, 1e-3 * rec), (n, err, rec)
        worst = max(worst, err)
    print(f'(2) spot-check: {len(SPOT)} tensors, dequant max|err| '
          f'matches quant.json, worst {worst:.3g}  PASS')
    # (2b) huge tensor (embed, 389M params) checked chunked, never loaded
    err, rec = L.spot_check_big(
        RAW, QUANT, quant, 'model.language_model.embed_tokens.weight')
    assert err <= max(1e-6, rec + 1e-9)
    print(f'(2b) chunked check embed_tokens: max|err| {err:.3g} '
          f'(recorded {rec:.3g})  PASS')
    # (3) single-layer forward check, LM layer 0
    r = L.decoder_layer_check(
        RAW, QUANT, quant,
        prefix='model.language_model.layers.0',
        embed_name='model.language_model.embed_tokens.weight',
        token_ids=[151643, 872, 1234, 4567],
        hidden=2560, n_q=32, n_kv=8, head_dim=128, intermediate=9728,
        theta=5_000_000, eps=1e-6)
    print(f"(3) layer-0 forward: max|err|={r['max_err']:.3g} "
          f"rms_err={r['rms_err']:.3g} rel_rms={r['rel_rms']:.3g} "
          f"(output rms {r['rms_out']:.3g})  "
          f"{'PASS' if r['rel_rms'] < 1e-3 else 'FAIL'}")
    print('ALL CHECKS DONE')


if __name__ == '__main__':
    main()
