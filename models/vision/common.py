#!/usr/bin/env python3
"""Shared utilities for Stage 6 model prep: fixed-point quantization and
numpy convolution matching the accelerator's data format.

Data format (accelerator):
  - pixels / unsigned activations: uint8, value = float * (255 / 2**r),
    r = per-layer activation range exponent (output covers [0, 2**r])
  - coefficients: int16, per-layer power-of-two scale  W_q = round(W * 2**e)
  - accumulator: 32-bit signed (golden uses int64 to avoid wrap; RTL tiles
    channels so the per-tile 32-bit accumulator stays in range)
  - after accumulation: arithmetic right shift, saturate to [0,255]
    (ReLU is free: the [0,255] saturation already clips negatives to 0)
  - bias: b_q = round(b * (255 / 2**r_in) * 2**e), added to the accumulator

Quantization scheme (uniform across all three models):
  per conv layer, e = floor(log2(32767 / max|W|))  (max precision in int16)
  W_q = clip(round(W * 2**e), -32768, 32767).astype(int16)
  RTL runtime shift for the layer = e + r_out - r_in, kept in [0,15] by
  reducing e when needed (slightly lower weight precision, shift stays legal)
  Signed intermediates (a layer with NO activation, e.g. RetinexNet conv0,
  DnCNN residual, RetinexNet recon logit): stored as SIGNED int8 / int32
  after the shift (no [0,255] clip). Zero padding inserts 0, which correctly
  represents 0.0 for a signed activation. Do NOT use a uint8+offset
  representation for a signed intermediate: it is incompatible with zero
  padding (zero-padded pixels carry no offset while the bias correction
  assumes all pixels do) -- verified on RetinexNet (conv1 error up to 215).
"""
import numpy as np


def range_exp(max_act):
    """Activation range exponent r for a layer output: the output is
    represented as uint8 = float * (255 / 2**r), covering [0, 2**r].
    r = 0 if max_act <= 1, else ceil(log2(max_act))."""
    import math
    if max_act <= 1.0:
        return 0
    return int(math.ceil(math.log2(max_act)))


def quantize_layer(W, b, r_in=0, r_out=0):
    """Quantize one conv layer's weights/bias.
    W: (H,W,Cin,Cout) float32, b: (Cout,) float32.
    r_in: range exponent of this layer's input activations
          (input uint8 = float * 255/2**r_in)
    r_out: range exponent of this layer's output activations
          (output uint8 = float * 255/2**r_out)
    Returns (W_q int16, b_q int32, shift int, scale float).
    Weight scale: W_q = round(W * 2**e), e = floor(log2(32767/max|W|)).
    RTL shift = e + r_out - r_in.  Bias: b_q = round(b * (255/2**r_in) * 2**e)."""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.abs(W).max() if W.size else 0.0
    if m <= 0:
        e = 0
    else:
        e = int(np.floor(np.log2(32767.0 / m)))
        e = max(0, min(15, e))
    # RTL shift = e + r_out - r_in must stay in [0,15]; reduce e if needed
    # (slightly lower weight precision, keeps the shift representable).
    e = min(e, 15 - (r_out - r_in))
    e = max(0, e)
    W_q = np.clip(np.round(W * (2.0 ** e)), -32768, 32767).astype(np.int16)
    s_in = 255.0 / (2.0 ** r_in)
    b_q = np.round(b * s_in * (2.0 ** e)).astype(np.int32)
    shift = e + r_out - r_in
    scale = 2.0 ** (-e)  # float weight = W_q * scale
    return W_q, b_q, shift, scale


def pad2d(x, pad, mode='zero'):
    """Pad (H,W,C) array. pad = (t,b,l,r) ints. mode 'zero' or 'replicate'."""
    t, b, l, r = pad
    if t == b == l == r == 0:
        return x
    if mode == 'zero':
        return np.pad(x, ((t, b), (l, r), (0, 0)), mode='constant')
    if mode == 'replicate':
        return np.pad(x, ((t, b), (l, r), (0, 0)), mode='edge')
    raise ValueError(mode)


def conv2d_float(x, W, b, pad=(0, 0, 0, 0), pad_mode='zero'):
    """Float conv. x:(H,W,Cin) float, W:(KH,KW,Cin,Cout) float, b:(Cout,).
    Returns (Hout,Wout,Cout) float. Valid conv after padding."""
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    xp = pad2d(x, pad, pad_mode)
    H, Wd, Cin = xp.shape
    KH, KW, _, Cout = W.shape
    Hout, Wout = H - KH + 1, Wd - KW + 1
    out = np.zeros((Hout, Wout, Cout), dtype=np.float64)
    # im2col-free direct loop over kernel taps (small test images only)
    for kh in range(KH):
        for kw in range(KW):
            # xp[kh:kh+Hout, kw:kw+Wout, :] @ W[kh,kw,:,:]
            out += np.tensordot(xp[kh:kh + Hout, kw:kw + Wout, :], W[kh, kw, :, :],
                                axes=([2], [0]))
    out += b
    return out


def conv2d_fixed(A, W_q, b_q, shift, pad=(0, 0, 0, 0), pad_mode='zero',
                 signed_out=False):
    """Fixed-point conv matching the RTL. A:(H,W,Cin) uint8 (0..255),
    representing float * (255/2**r_in).
    W_q:(KH,KW,Cin,Cout) int16, b_q:(Cout,) int32, shift: RTL right shift.
    Output uint8 = float_out * (255/2**r_out) if signed_out False
    (clip [0,255]); else signed int array after shift (no clip).
    Accumulation in int64."""
    A = np.asarray(A)
    Wq = np.asarray(W_q, dtype=np.int64)
    bq = np.asarray(b_q, dtype=np.int64)
    Ap = pad2d(A, pad, pad_mode)
    H, Wd, Cin = Ap.shape
    KH, KW, _, Cout = Wq.shape
    Hout, Wout = H - KH + 1, Wd - KW + 1
    acc = np.zeros((Hout, Wout, Cout), dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            acc += np.tensordot(Ap[kh:kh + Hout, kw:kw + Wout, :].astype(np.int64),
                                Wq[kh, kw, :, :], axes=([2], [0]))
    acc += bq
    # arithmetic right shift (floor division for negatives)
    out = np.floor_divide(acc, 2 ** shift)
    if signed_out:
        return out
    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(a, b, maxval=255.0):
    """PSNR between two arrays (same shape). a,b in [0,maxval] scale."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(maxval * maxval / mse)


def max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) -
                               np.asarray(b, dtype=np.float64))))


def make_test_gray(h=32, w=32, seed=7):
    """Deterministic synthetic grayscale test image, uint8 (0..255).
    Smooth gradient + a bright disk + a dark bar + fine texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 90.0 + 60.0 * (xx / max(w - 1, 1)) + 30.0 * np.sin(yy / 5.0)
    # bright disk
    img += 70.0 * ((xx - w * 0.65) ** 2 + (yy - h * 0.35) ** 2 < (min(h, w) * 0.22) ** 2)
    # dark bar
    img -= 55.0 * ((np.abs(yy - h * 0.7) < 2) & (xx > w * 0.2) & (xx < w * 0.8))
    # fine texture
    img += 12.0 * rng.standard_normal((h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_test_rgb(h=32, w=32, seed=7):
    """Deterministic synthetic RGB test image, uint8 (H,W,3)."""
    g = make_test_gray(h, w, seed).astype(np.float64)
    r = np.clip(g * 1.05 + 10 * np.sin(np.mgrid[0:h, 0:w][1] / 4.0), 0, 255)
    b = np.clip(g * 0.85 + 8 * np.cos(np.mgrid[0:h, 0:w][0] / 6.0), 0, 255)
    return np.stack([r, g, b], axis=2).astype(np.uint8)
