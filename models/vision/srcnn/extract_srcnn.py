#!/usr/bin/env python3
"""Extract the original SRCNN 9-1-5 (x3) author weights from the FFmpeg
GSoC patch email (mail-archive msg67144), which embeds the author's
srcnn_weights.h verbatim. Parses the 6 float arrays from the HTML.
Layout (from ff_cnn_num_filters = {1,64,32,1}, ff_cnn_size_filters = {1,9,1,5}):
  W1: 64 filters of 9x9x1  (weights1_64x81, row-major per filter)
  W2: 32 filters of 1x1x64 (weights2_32x64x1)
  W3:  1 filter  of 5x5x32 (weights3_32x25)
  b1: 64, b2: 32, b3: 1
Output arrays are (H, W, Cin, Cout) float32."""
import re, numpy as np, html

raw = open('raw/msg67144.html', encoding='utf-8', errors='replace').read()
# strip HTML tags to get plain text (numbers are plain text in the patch)
text = re.sub(r'<[^>]+>', ' ', raw)
text = html.unescape(text)

def grab(name):
    # find "name[] = {" then collect floats until "};"
    m = re.search(re.escape(name) + r'\[\]\s*=\s*\{', text)
    assert m, f"array {name} not found"
    start = m.end()
    end = text.find('};', start)
    assert end > 0, f"terminator for {name} not found"
    chunk = text[start:end]
    # remove the leading '+' diff markers and any stray words
    chunk = chunk.replace('+', ' ')
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', chunk)
    vals = np.array([float(x) for x in nums], dtype=np.float32)
    return vals

w1 = grab('weights1_64x81')
w2 = grab('weights2_32x64x1')
w3 = grab('weights3_32x25')
b1 = grab('biases1_64')
b2 = grab('biases2_32')
b3 = grab('biases3_1')
print("counts:", len(w1), len(w2), len(w3), len(b1), len(b2), len(b3))
assert len(w1) == 64*81 and len(w2) == 32*64 and len(w3) == 32*25
assert len(b1) == 64 and len(b2) == 32 and len(b3) == 1

# reshape to (H, W, Cin, Cout)
W1 = w1.reshape(64, 9, 9).transpose(1, 2, 0).reshape(9, 9, 1, 64)  # per-filter 9x9, 1 in-ch
# weights1 is stored filter-major: filter f occupies w1[f*81:(f+1)*81] in row-major 9x9
W1 = w1.reshape(64, 9, 9).transpose(1, 2, 0)  # (9,9,64) -> need (9,9,1,64)
W1 = W1[:, :, None, :]
W2 = w2.reshape(32, 64).transpose(1, 0).reshape(1, 1, 64, 32)  # (1,1,64,32)
W3 = w3.reshape(32, 5, 5).transpose(1, 2, 0).reshape(5, 5, 32, 1)  # (5,5,32,1)

for nm, W, b in [('L1', W1, b1), ('L2', W2, b2), ('L3', W3, b3)]:
    print(f"  {nm}: W {W.shape} range [{W.min():+.4f},{W.max():+.4f}] |W|mean {np.abs(W).mean():.4f}  "
          f"b {b.shape} range [{b.min():+.4f},{b.max():+.4f}]")

np.savez('srcnn_x3_float.npz', L1_W=W1, L1_b=b1, L2_W=W2, L2_b=b2, L3_W=W3, L3_b=b3)
print("saved srcnn_x3_float.npz")
