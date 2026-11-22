#!/usr/bin/env python3
"""Shared safetensors reader + repo-scheme int16 quantizer for the LLM
models (models/llm/). No safetensors / transformers dependency: the
safetensors format is an 8-byte little-endian header length, a JSON
header, then raw little-endian tensor bytes, so plain struct + json +
numpy memmap is enough.

Memory rule (VM has ~6 GB, ~0.5 GB free at run time): NEVER load a
whole tensor. Everything here streams in chunks (default 4M elements
per chunk) through np.memmap; BF16 is converted to float32 chunk-wise
with the exact bit trick (uint16 << 16 viewed as float32).

Quantization scheme (same as models/vision/common.py quantize_layer,
adapted — see the differences noted in each model README):
  per tensor, e = floor(log2(32767 / max|W|)), clamped to [0, 15],
  W_q = clip(round(W * 2**e), -32768, 32767).astype(int16),
  float weight = W_q * 2**(-e).
Differences vs the vision flow: no activation calibration exists for
these models (no forward on real data), so there is no r_in / r_out
and no RTL shift; every tensor — weight, bias, norm, embedding,
lm_head — is quantized on its own per-tensor scale, and the scale is
recorded in quant.json. Biases are NOT folded with an activation
scale (there is none); they are quantized as standalone tensors.
"""
import json
import math
import os
import struct

import numpy as np

CHUNK_ELEMS = 2_000_000  # elements per streaming chunk (~8 MB fp32)


def read_st_header(path):
    """Read a .safetensors header (dict name -> {dtype, shape,
    data_offsets}). Does not touch tensor data."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def load_index(raw_dir):
    with open(os.path.join(raw_dir, 'model.safetensors.index.json')) as f:
        return json.load(f)['weight_map']


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _bf16_chunk(u16):
    """Exact BF16 -> float32 for a uint16 chunk (little-endian)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def iter_f32_chunks(mm, dtype, shape, data_off, data_start):
    """Yield float32 chunks of one tensor from a memmap of the shard
    file. mm is a np.memmap of the whole file (uint8); data_start is
    the byte offset where tensor data begins in the file."""
    n = numel(shape)
    start, end = data_off
    if dtype == 'BF16':
        total_bytes = end - start
        assert total_bytes == n * 2, (shape, total_bytes, n)
        pos = 0
        while pos < n:
            take = min(CHUNK_ELEMS, n - pos)
            u16 = np.frombuffer(mm, dtype='<u2', count=take,
                                offset=data_start + start + pos * 2)
            yield _bf16_chunk(u16)
            pos += take
    elif dtype == 'F32':
        assert end - start == n * 4, (shape, end - start, n)
        pos = 0
        while pos < n:
            take = min(CHUNK_ELEMS, n - pos)
            yield np.frombuffer(mm, dtype='<f4', count=take,
                                offset=data_start + start + pos * 4).astype(np.float32)
            pos += take
    elif dtype == 'F64':
        pos = 0
        while pos < n:
            take = min(CHUNK_ELEMS, n - pos)
            yield np.frombuffer(mm, dtype='<f8', count=take,
                                offset=data_start + start + pos * 8).astype(np.float32)
            pos += take
    else:
        raise ValueError(f'unsupported dtype {dtype}')


def tensor_maxabs(mm, dtype, shape, data_off, data_start):
    m = 0.0
    for c in iter_f32_chunks(mm, dtype, shape, data_off, data_start):
        if c.size:
            m = max(m, float(np.max(np.abs(c))))
    return m


def quantize_e(max_abs):
    """Repo scheme: e = floor(log2(32767 / max|W|)), clamped [0, 15]."""
    if max_abs <= 0:
        return 0
    e = int(math.floor(math.log2(32767.0 / max_abs)))
    return max(0, min(15, e))


def quantize_tensor(mm, dtype, shape, data_off, data_start, out_f):
    """Two-pass streaming quantize of one tensor. Writes int16 bytes
    to out_f (already positioned). Returns stats dict."""
    n = numel(shape)
    max_abs = tensor_maxabs(mm, dtype, shape, data_off, data_start)
    e = quantize_e(max_abs)
    scale = 2.0 ** (-e)
    mult = 2.0 ** e
    max_err = 0.0
    sum_sq_err = 0.0
    sum_sq_sig = 0.0
    clipped = 0
    for c in iter_f32_chunks(mm, dtype, shape, data_off, data_start):
        q = np.round(c.astype(np.float64) * mult)
        clipped += int(np.sum(np.abs(q) > 32767))
        q = np.clip(q, -32768, 32767).astype(np.int16)
        deq = q.astype(np.float64) * scale
        err = c.astype(np.float64) - deq
        if err.size:
            max_err = max(max_err, float(np.max(np.abs(err))))
            sum_sq_err += float(np.sum(err * err))
            sum_sq_sig += float(np.sum(c.astype(np.float64) ** 2))
        out_f.write(q.tobytes(order='C'))
    rms_err = math.sqrt(sum_sq_err / n) if n else 0.0
    rms_sig = math.sqrt(sum_sq_sig / n) if n else 0.0
    snr_db = (20.0 * math.log10(rms_sig / rms_err)
              if rms_err > 0 and rms_sig > 0 else float('inf'))
    return {'shape': list(shape), 'dtype_orig': dtype, 'numel': n,
            'e': e, 'scale': scale, 'max_abs': max_abs,
            'max_err': max_err, 'rms_err': rms_err,
            'rms_signal': rms_sig, 'snr_db': snr_db,
            'clipped': clipped}


def open_shard(path):
    """Return (header, memmap, data_start) for a .safetensors file."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    data_start = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    return header, mm, data_start


_QUANT_CACHE = {}


def dequant_tensor(quant_dir, index, name, scale=None):
    """Load ONE quantized tensor, dequantized to float32 (small
    tensors only — callers must not use this for the embedding).
    scale defaults to the per-tensor scale from <model>/quant.json."""
    if scale is None:
        qpath = os.path.join(os.path.dirname(quant_dir), 'quant.json')
        if qpath not in _QUANT_CACHE:
            with open(qpath) as f:
                _QUANT_CACHE[qpath] = json.load(f)['tensors']
        scale = _QUANT_CACHE[qpath][name]['scale']
    shard = index[name]
    header, mm, data_start = open_shard(os.path.join(quant_dir, shard))
    meta = header[name]
    shape = meta['shape']
    start, end = meta['data_offsets']
    n = numel(shape)
    raw = np.frombuffer(mm, dtype='<i2', count=n,
                        offset=data_start + start).astype(np.float32)
    return (raw * scale).reshape(shape)


def quantize_model(raw_dir, quant_dir, model_name):
    """Quantize every tensor of a sharded safetensors checkpoint.
    Mirrors the raw shard layout under quant_dir (same file names,
    same tensor->shard map, dtype I16). Writes quant.json (per-tensor
    stats) and model.safetensors.index.json. Returns the quant dict."""
    import time
    os.makedirs(quant_dir, exist_ok=True)
    index = load_index(raw_dir)
    shards = sorted(set(index.values()))
    by_shard = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)
    quant = {'model': model_name, 'scheme': 'int16 per-tensor power-of-two scale, '
             'e = floor(log2(32767/max|W|)) clamped [0,15], W_q = round(W*2**e)',
             'tensors': {}}
    t0 = time.time()
    total = 0
    for shard in shards:
        raw_path = os.path.join(raw_dir, shard)
        header, mm, data_start = open_shard(raw_path)
        # output header: same names/shapes, dtype I16, recomputed offsets
        out_header = {}
        off = 0
        for name in sorted(by_shard[shard]):
            meta = header[name]
            n = numel(meta['shape'])
            out_header[name] = {'dtype': 'I16', 'shape': meta['shape'],
                                'data_offsets': [off, off + n * 2]}
            off += n * 2
        out_path = os.path.join(quant_dir, shard)
        with open(out_path, 'wb') as out_f:
            blob = json.dumps(out_header, separators=(',', ':')).encode()
            out_f.write(struct.pack('<Q', len(blob)))
            out_f.write(blob)
            for name in sorted(by_shard[shard]):
                meta = header[name]
                stats = quantize_tensor(mm, meta['dtype'], meta['shape'],
                                        meta['data_offsets'], data_start, out_f)
                quant['tensors'][name] = stats
                total += 1
                if total % 500 == 0:
                    print(f'  {total}/{len(index)} tensors, '
                          f'{time.time() - t0:.0f}s', flush=True)
        print(f'  shard {shard}: {len(by_shard[shard])} tensors -> {out_path}',
              flush=True)
    # index for the quantized tree (same weight map, int16 total size)
    total_bytes = sum(s['numel'] * 2 for s in quant['tensors'].values())
    with open(os.path.join(quant_dir, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': total_bytes},
                   'weight_map': index}, f, indent=2, sort_keys=True)
    # quant.json lives in the model dir (tracked), NOT in quantized/
    # (which is gitignored): the scales are the registration record.
    with open(os.path.join(os.path.dirname(quant_dir), 'quant.json'), 'w') as f:
        json.dump(quant, f, indent=1, sort_keys=True)
    print(f'  done: {total} tensors, {total_bytes / 2**30:.2f} GiB int16, '
          f'{time.time() - t0:.0f}s')
    return quant


def embedding_rows(raw_dir, index, embed_name, ids, quant_dir=None,
                   quant=None):
    """Read a few embedding rows (token ids) as float32, from raw
    (BF16) or from quantized (int16 + scale from quant.json)."""
    import gc
    shard = index[embed_name]
    src_dir = quant_dir if quant_dir else raw_dir
    header, mm, data_start = open_shard(os.path.join(src_dir, shard))
    meta = header[embed_name]
    shape = meta['shape']
    rows, cols = shape
    start, _ = meta['data_offsets']
    out = np.empty((len(ids), cols), dtype=np.float32)
    if quant_dir:
        scale = quant['tensors'][embed_name]['scale']
        for i, t in enumerate(ids):
            q = np.frombuffer(mm, dtype='<i2', count=cols,
                              offset=data_start + start + t * cols * 2)
            out[i] = q.astype(np.float32) * scale
    else:
        for i, t in enumerate(ids):
            u16 = np.frombuffer(mm, dtype='<u2', count=cols,
                                offset=data_start + start + t * cols * 2)
            out[i] = _bf16_chunk(u16)
    del mm
    gc.collect()
    return out


def _rope(x, theta, positions):
    """Standard RoPE on (seq, n_heads, head_dim). x is float64.
    (The Qwen models use mRoPE; for text-only positions the check
    uses plain RoPE — the identical transform is applied in the float
    and the dequantized pass, so the quantization-error measurement
    is unaffected.)"""
    seq, nh, d = x.shape
    half = d // 2
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
    ang = np.outer(np.asarray(positions, dtype=np.float64), inv_freq)
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def decoder_layer_check(raw_dir, quant_dir, quant, prefix, embed_name,
                        token_ids, hidden, n_q, n_kv, head_dim,
                        intermediate, theta, eps=1e-6):
    """One decoder layer, float (raw) vs dequantized int16, same input.
    Returns dict with max|err|, RMS err, relative RMS, output RMS.
    Streams one weight at a time; peak ~300 MB (largest MLP tensor
    in float64)."""
    import gc
    index = load_index(raw_dir)
    qindex = load_index(quant_dir)
    assert index == qindex, 'quantized index weight map differs from raw'

    def w_raw(name):
        a = load_f32_tensor(raw_dir, index, name).astype(np.float64)
        return a

    def w_deq(name):
        q = dequant_tensor(quant_dir, qindex, name)
        return q.astype(np.float64)

    def layer(x, get_w):
        # input layernorm
        w = get_w(prefix + '.input_layernorm.weight')
        h = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w
        del w; gc.collect()
        seq = x.shape[0]
        q = h @ get_w(prefix + '.self_attn.q_proj.weight').T
        k = h @ get_w(prefix + '.self_attn.k_proj.weight').T
        v = h @ get_w(prefix + '.self_attn.v_proj.weight').T
        q = q.reshape(seq, n_q, head_dim)
        k = k.reshape(seq, n_kv, head_dim)
        v = v.reshape(seq, n_kv, head_dim)
        pos = np.arange(seq)
        q = _rope(q, theta, pos)
        k = _rope(k, theta, pos)
        rep = n_q // n_kv
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
        # attention over sequence: (heads, seq, dim)
        qt = q.transpose(1, 0, 2)
        kt = k.transpose(1, 0, 2)
        vt = v.transpose(1, 0, 2)
        att = np.matmul(qt, kt.transpose(0, 2, 1)) / np.sqrt(head_dim)
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        att = np.where(mask[None, :, :], -1e30, att)
        att = att - att.max(axis=-1, keepdims=True)
        p = np.exp(att)
        p = p / p.sum(axis=-1, keepdims=True)
        o = np.matmul(p, vt).transpose(1, 0, 2).reshape(seq, n_q * head_dim)
        o = o @ get_w(prefix + '.self_attn.o_proj.weight').T
        x = x + o
        del q, k, v, o, att, p; gc.collect()
        w = get_w(prefix + '.post_attention_layernorm.weight')
        h = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w
        del w; gc.collect()
        g = h @ get_w(prefix + '.mlp.gate_proj.weight').T
        u = h @ get_w(prefix + '.mlp.up_proj.weight').T
        g = g / (1.0 + np.exp(-g))  # SiLU
        d = (g * u) @ get_w(prefix + '.mlp.down_proj.weight').T
        del g, u; gc.collect()
        return x + d

    x0 = embedding_rows(raw_dir, index, embed_name, token_ids).astype(np.float64)
    out_f = layer(x0, w_raw)
    del x0; gc.collect()
    x0q = embedding_rows(raw_dir, index, embed_name, token_ids,
                         quant_dir=quant_dir, quant=quant).astype(np.float64)
    out_q = layer(x0q, w_deq)
    err = out_f - out_q
    return {'max_err': float(np.max(np.abs(err))),
            'rms_err': float(np.sqrt(np.mean(err * err))),
            'rms_out': float(np.sqrt(np.mean(out_f * out_f))),
            'rel_rms': float(np.sqrt(np.mean(err * err)) /
                             np.sqrt(np.mean(out_f * out_f)))}


def spot_check_big(raw_dir, quant_dir, quant, name, n_elems=2_000_000):
    """Chunked dequant check for a huge tensor (embed / lm_head):
    compare the first n_elems only, never materializing the tensor.
    Returns (max_err, recorded max_err)."""
    index = load_index(raw_dir)
    qindex = load_index(quant_dir)
    scale = quant['tensors'][name]['scale']
    shard = index[name]
    header, mm, data_start = open_shard(os.path.join(raw_dir, shard))
    meta = header[name]
    qheader, qmm, qdata_start = open_shard(os.path.join(quant_dir, shard))
    qmeta = qheader[name]
    max_err = 0.0
    seen = 0
    for c in iter_f32_chunks(mm, meta['dtype'], meta['shape'],
                             meta['data_offsets'], data_start):
        take = min(c.size, n_elems - seen)
        if take <= 0:
            break
        q = np.frombuffer(qmm, dtype='<i2', count=take,
                          offset=qdata_start + qmeta['data_offsets'][0] + seen * 2)
        err = np.abs(c[:take].astype(np.float64) -
                     q.astype(np.float64) * scale)
        if err.size:
            max_err = max(max_err, float(np.max(err)))
        seen += take
        if seen >= n_elems:
            break
    return max_err, quant['tensors'][name]['max_err']


def load_f32_tensor(raw_dir, index, name):
    """Load ONE raw tensor as float32 (small tensors only)."""
    shard = index[name]
    header, mm, data_start = open_shard(os.path.join(raw_dir, shard))
    meta = header[name]
    shape = meta['shape']
    out = np.empty(numel(shape), dtype=np.float32)
    pos = 0
    for c in iter_f32_chunks(mm, meta['dtype'], shape,
                             meta['data_offsets'], data_start):
        out[pos:pos + c.size] = c
        pos += c.size
    return out.reshape(shape)
