# Qwen3-VL-4B-Instruct (raw weights, download only)

- Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- License: Apache 2.0
- Fetched: 2026-11-21 (PST), via `bash models/llm/fetch_llm.sh` (repo root)
- Raw dir: `models/llm/qwen3-vl-4b/raw/` — total 8.28 GiB (8,887,292,732 bytes)
- Architecture: 36-layer text decoder (hidden 2560, SwiGLU 9728, 4.4B
  params) + 24-layer ViT (hidden 1024, DeepStack); 256k context.
- Status: DOWNLOAD ONLY — hash-verified against the Hub (both safetensors
  shards match their LFS SHA256), not loaded / quantized / registered.
  Raw weights are gitignored (`models/.gitignore`: `llm/*/raw/`).

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
