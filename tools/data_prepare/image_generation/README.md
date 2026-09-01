# BLIP3o image generation

`convert_blip3o_keys.py` rebuilds image-generation training JSONL from the released
canonical key lists and the official BLIP3o tar files. A key has this form:

```text
BLIP3o/BLIP3o-Pretrain-Short-Caption/00000/000000524.jpg
BLIP3o/BLIP3o-Pretrain-Long-Caption/sa_000912/sa_10204894.jpg
```

Long Caption and Long part2 both use the
`BLIP3o-Pretrain-Long-Caption` namespace. Key-list row order is retained in
the generated JSONL. Image paths in the JSONL are normalized to
`<tar stem>/<image member>`, and `id`/`index` are reassigned as zero-based
output row numbers.

```bash
# BLIP3o-Pretrain-Short-Caption.jsonl
python tools/data_prepare/image_generation/convert_blip3o_keys.py \
  --blip3o-root datas/train_data/BLIP3o \
  --key-list jsonl_generate/train_jsonls/image_generation/keys/BLIP3o-Pretrain-Short-Caption.keys.txt.gz \
  --jsonl-out-path jsonl_generate/train_jsonls/image_generation/BLIP3o-Pretrain-Short-Caption.jsonl

# BLIP3o-Pretrain-Long-Caption.jsonl
python tools/data_prepare/image_generation/convert_blip3o_keys.py \
  --blip3o-root datas/train_data/BLIP3o \
  --key-list jsonl_generate/train_jsonls/image_generation/keys/BLIP3o-Pretrain-Long-Caption.keys.txt.gz \
  --jsonl-out-path jsonl_generate/train_jsonls/image_generation/BLIP3o-Pretrain-Long-Caption.jsonl

# BLIP3o-Pretrain-Long-Caption-part2.jsonl
python tools/data_prepare/image_generation/convert_blip3o_keys.py \
  --blip3o-root datas/train_data/BLIP3o \
  --key-list jsonl_generate/train_jsonls/image_generation/keys/BLIP3o-Pretrain-Long-Caption-part2.keys.txt.gz \
  --jsonl-out-path jsonl_generate/train_jsonls/image_generation/BLIP3o-Pretrain-Long-Caption-part2.jsonl
```

`--key-list` defaults to
`jsonl_generate/train_jsonls/image_generation/keys/BLIP3o-Pretrain-Long-Caption.keys.txt.gz`,
and `--jsonl-out-path` defaults to the corresponding JSONL under
`jsonl_generate/train_jsonls/image_generation/`. The converter uses a temporary
SQLite index so each tar is opened once while the output still follows key-list
order. Use `--work-dir` to place that index on a disk with enough free space.
