# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import pdb
import os
import yaml
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from modeling.qwen2 import Qwen2Tokenizer
from pretrain_unified_navit import ModelArguments, DataArguments, TrainingArguments
import pandas as pd


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.results_dir, exist_ok=True)

    # --- tokenizer ---
    tokenizer = Qwen2Tokenizer.from_pretrained(
        training_args.resume_from_hf if training_args.resume_from_hf else model_args.llm_path
    )
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

    # --- dataset config ---
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)

    dataset_config = DataConfig(grouped_datasets=dataset_meta)

    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        # Set a default because vae_image_downsample may be absent from the dataset config.
        dataset_config.vae_image_downsample = model_args.latent_patch_size * 8
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    # --- dataset class ---
    dataset_class = PackedDataset
    collate_fn = collate_wrapper()

    train_dataset = dataset_class(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=0,
        world_size=1,
        num_workers=0,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=None,
    )

    train_dataset.set_epoch(data_args.data_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # packed dataset = 1
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --- test loop ---
    sample_status = list()
    tokens_status = list()
    for curr_step, batch in enumerate(train_loader):
        print(f"\n=== Batch {curr_step} ===")
        print(
            f"sequence_length={batch.sequence_length}, num_samples_in_packed={len(batch.sample_lens)}"
        )

        data = batch.to_dict()
        grouped_names = data.get("grouped_names", None)
        batch_group_name = data.get("batch_group_name", None)

        total_samples = len(data['sample_lens'])
        total_tokens = data['sequence_length']
        sample_lens = data['sample_lens']

        batch_group_counts = {'step': curr_step}
        batch_group_tokens = {'step': curr_step}
        for group_name in grouped_names:
            batch_group_counts.update({group_name: 0})
            batch_group_tokens.update({group_name: 0})
        batch_group_counts.update({'total_samples': total_samples})
        batch_group_tokens.update({'total_tokens': total_tokens})
        for sample_name, sample_len in zip(batch_group_name, sample_lens):
            batch_group_counts[sample_name] += 1
            batch_group_tokens[sample_name] += sample_len

        sample_status.append(batch_group_counts)
        tokens_status.append(batch_group_tokens)
        if curr_step >= training_args.total_steps:
            break
    sample_df = pd.DataFrame(sample_status)
    tokens_df = pd.DataFrame(tokens_status)

    sample_df.to_csv(os.path.join(training_args.results_dir, 'sample_status.csv'))
    tokens_df.to_csv(os.path.join(training_args.results_dir, 'tokens_status.csv'))


if __name__ == "__main__":
    main()
