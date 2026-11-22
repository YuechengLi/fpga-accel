# Qwen2.5-Omni-3B (raw weights, download only)

> **LICENSE: RESEARCH ONLY.** This model is released under the Qwen
> Research License Agreement for **non-commercial use only**. Commercial
> use requires a separate license from Alibaba's Qwen Team. Research use
> (this repo's purpose) is permitted.

- Source: https://huggingface.co/Qwen/Qwen2.5-Omni-3B
- Fetched: 2026-11-21 (PST), via `bash models/llm/fetch_llm.sh` (repo root)
- Raw dir: `models/llm/qwen2.5-omni-3b/raw/` — total 11.17 GiB (11,989,103,936 bytes)
- Architecture: Thinker-Talker (3B dense Thinker + Talker speech decoder +
  vision / audio encoders); text, image, audio, video in; text + speech out.
- Note: the full checkpoint is ~11.2 GiB BF16 (the "3B" counts the Thinker;
  Talker + encoders add the rest).
- Status: DOWNLOAD ONLY — hash-verified against the Hub (all 5 LFS files
  match their LFS SHA256), not loaded / quantized / registered.
  Raw weights are gitignored (`models/.gitignore`: `llm/*/raw/`).

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
