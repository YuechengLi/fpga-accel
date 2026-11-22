#!/bin/bash
# Download-only fetch of Qwen3-VL-4B-Instruct + Qwen2.5-Omni-3B from
# Hugging Face into models/llm/*/raw/ (re-runnable: curl resumes with -C -).
# No checkpoint is loaded here; verification + quantization + registration
# is a separate step. Usage: bash models/llm/fetch_llm.sh
# Requires: curl, python3 (stdlib only). No huggingface_hub needed.
set -u
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 - "$REPO_DIR" <<'PYEOF'
import json, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

base = sys.argv[1]
repos = [
    ("Qwen/Qwen3-VL-4B-Instruct", "qwen3-vl-4b"),
    ("Qwen/Qwen2.5-Omni-3B", "qwen2.5-omni-3b"),
]
jobs = []  # (repo, path, size, out, is_shard)
for repo, sub in repos:
    dest = os.path.join(base, sub, "raw")
    os.makedirs(dest, exist_ok=True)
    api = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(api) as r:
        data = json.load(r)
    for e in data:
        if e.get("type") != "file":
            continue
        path, size = e["path"], e.get("size", 0)
        out = os.path.join(dest, path)
        os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
        jobs.append((repo, path, size, out, path.endswith(".safetensors")))

def fetch(job):
    repo, path, size, out, _ = job
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    r = subprocess.run(["curl", "-fSL", "--retry", "3", "--retry-delay", "5",
                        "-C", "-", "-o", out, url],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return (repo, path, f"curl rc={r.returncode}")
    got = os.path.getsize(out)
    if got != size:
        return (repo, path, f"size mismatch: got {got}, expected {size}")
    print(f"OK {repo} {path} ({got} bytes)", flush=True)
    return None

small = [j for j in jobs if not j[4]]
shards = [j for j in jobs if j[4]]
fails = []
print(f"small files: {len(small)}, shards: {len(shards)}", flush=True)
for j in small:
    f = fetch(j)
    if f:
        fails.append(f)
with ThreadPoolExecutor(max_workers=5) as ex:
    for f in ex.map(fetch, shards):
        if f:
            fails.append(f)
if fails:
    print("FAILURES:", flush=True)
    for repo, path, why in fails:
        print(f"  {repo} {path}: {why}", flush=True)
    sys.exit(1)
print("FETCH COMPLETE", flush=True)
PYEOF
