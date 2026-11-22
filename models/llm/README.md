# LLM models (download only)

Large multimodal models for the VCU128 track (see `docs/vcu128-bringup.md`).
Fetched 2026-11-21 (PST) from Hugging Face with `bash models/llm/fetch_llm.sh`
(run from the repo root; re-runnable, curl resumes with `-C -`).

**Status: DOWNLOAD ONLY.** The raw weights below have been hash-verified
against the Hugging Face Hub (all LFS SHA256 match; all other files match
their Hub sizes), but they have NOT been loaded, quantized, or registered
in the repo. Verification + quantization + registration is a separate step
that needs explicit approval (same pattern as the YOLOv8n / DNANet
onboarding in `models/`).

**Raw weights are NEVER committed:** `models/.gitignore` ignores
`llm/*/raw/`. Only the READMEs, the fetch script, and the SHA256 manifests
in this directory are tracked.

## Qwen3-VL-4B-Instruct

- Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- License: Apache 2.0
- Architecture: 36-layer text decoder (hidden 2560, SwiGLU 9728, 4.4B
  params) + 24-layer ViT (hidden 1024, DeepStack); 256k context.
- Raw dir: `models/llm/qwen3-vl-4b/raw/` — total 8.28 GiB (8,887,292,732 bytes)

## Qwen2.5-Omni-3B

- Source: https://huggingface.co/Qwen/Qwen2.5-Omni-3B
- License: **RESEARCH ONLY** (Qwen Research License Agreement —
  non-commercial use; a separate license is needed for commercial use)
- Architecture: Thinker-Talker (3B dense Thinker + Talker speech decoder +
  vision / audio encoders); text, image, audio, video in; text + speech out.
- Raw dir: `models/llm/qwen2.5-omni-3b/raw/` — total 11.17 GiB (11,989,103,936 bytes)

## File manifest (SHA256)

### qwen3-vl-4b/raw/

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

### qwen2.5-omni-3b/raw/

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
