#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Held-out validation loss measured inside a training run.

`lerobot_train.py` only evaluates through `cfg.env`, so an offline run has no held-out signal until it
finishes and `lerobot_validate_offline` is run over its checkpoints. This module builds the same
measurement as that script, but against the live policy, so the curve is visible while training runs.

The frame subsample and the noise/time draws are taken from `lerobot_validate_offline` rather than
reimplemented, so the numbers logged during training are directly comparable with a later run of that
script over the same checkpoints. That is why this module imports from a script: keeping the two in
lockstep matters more here than the layering.
"""

import copy
import csv
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset

from src.lerobot.configs.train import TrainPipelineConfig
from src.lerobot.datasets.factory import make_dataset
from src.lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from src.lerobot.scripts.lerobot_validate_offline import parse_episodes, subsample_indices
from src.lerobot.utils.random_utils import get_rng_state, set_rng_state


def _resolve_val_dataset_cfg(cfg: TrainPipelineConfig) -> tuple[TrainPipelineConfig, list[int]]:
    """Copy of `cfg` whose dataset points at the held-out episodes, and those episodes.

    `cfg` itself must stay untouched: `save_checkpoint` writes the live config object into every
    `train_config.json` (`train.py:_save_pretrained`), so editing `cfg.dataset` here would leak the
    validation root into the run's own saved config.
    """
    ocfg = cfg.offline_eval
    val_root = Path(ocfg.dataset_root).expanduser()
    if not val_root.exists():
        # Not just a nicety: `LeRobotDatasetMetadata` creates a missing root and then pulls the repo
        # from the Hub, so a typo would silently train a val set out of whatever `repo_id` resolves to.
        raise FileNotFoundError(f"offline_eval.dataset_root does not exist: {val_root}")

    val_cfg = copy.deepcopy(cfg)
    val_cfg.dataset.root = str(val_root)
    val_cfg.dataset.repo_id = ocfg.dataset_repo_id or cfg.dataset.repo_id
    # Augmentation on held-out frames only adds variance; `lerobot_validate_offline` disables it too.
    val_cfg.dataset.image_transforms.enable = False

    meta = LeRobotDatasetMetadata(val_cfg.dataset.repo_id, root=val_cfg.dataset.root)
    available = list(meta.episodes["episode_index"])
    episodes = parse_episodes(ocfg.val_episodes, available)

    if cfg.dataset.root and Path(cfg.dataset.root).expanduser().resolve() == val_root.resolve():
        # Same dataset on both sides: anything scoped to the training episodes is contaminated, and
        # `cfg.dataset.episodes=None` means every episode was trained on.
        train_episodes = set(cfg.dataset.episodes if cfg.dataset.episodes is not None else available)
        overlap = sorted(set(episodes) & train_episodes)
        if overlap:
            raise ValueError(
                f"offline_eval scores {len(overlap)} episode(s) that the training dataset also contains "
                f"(e.g. {overlap[:5]}). Point --offline_eval.dataset_root at a dataset whose episodes are "
                "unseen, or pick a disjoint --offline_eval.val_episodes range."
            )

    val_cfg.dataset.episodes = episodes
    return val_cfg, episodes


def build_offline_val_loader(
    cfg: TrainPipelineConfig, device: torch.device
) -> tuple[DataLoader, dict[str, Any]]:
    """Fixed set of held-out frames, plus a loader that yields them in a stable order.

    Built once before the training loop so a bad path or episode range fails at startup rather than
    thousands of steps in, and so every evaluation scores the same frames.
    """
    val_cfg, episodes = _resolve_val_dataset_cfg(cfg)
    ocfg = cfg.offline_eval

    dataset = make_dataset(val_cfg)
    indices = subsample_indices(len(dataset), ocfg.max_samples, ocfg.seed)

    if ocfg.num_workers == 0:
        # Decoding video in the main process while the training loader's workers are decoding the same
        # files is documented to segfault (see the note on `LeRobotDataset._query_videos`).
        logging.warning(
            "offline_eval.num_workers=0 decodes video in the main process, which "
            "`LeRobotDataset._query_videos` warns causes a Segmentation Fault alongside the training "
            "dataloader's own workers. Consider --offline_eval.num_workers>0."
        )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=ocfg.batch_size,
        # shuffle=False keeps the batch_index -> frames mapping fixed, which the per-batch noise seed
        # below depends on to make checkpoints comparable.
        shuffle=False,
        num_workers=ocfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if ocfg.num_workers > 0 else None,
    )

    info = {
        "root": val_cfg.dataset.root,
        "repo_id": val_cfg.dataset.repo_id,
        "episodes": episodes,
        "frames": len(indices),
    }
    return loader, info


@contextmanager
def frozen_rng(seed: int) -> Generator[None, None, None]:
    """Run a block on a freshly seeded RNG, restoring the caller's state on the way out.

    Two reasons this is not just `torch.manual_seed`. Every checkpoint must see the *same* noise and
    time draws, otherwise the run-to-run spread in the loss swamps the difference being measured. And
    the draws must not leak back into training: a run with offline eval on would otherwise take a
    different trajectory from one with it off, which breaks single-variable comparisons.
    """
    saved = get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        set_rng_state(saved)


@torch.no_grad()
def compute_offline_val_loss(
    policy,
    preprocessor,
    loader: DataLoader,
    accelerator: Accelerator,
    seed: int,
) -> tuple[float, float, int]:
    """Mean and std of the flow-matching loss over the held-out frames.

    Mirrors `lerobot_validate_offline.evaluate_checkpoint`: same per-batch reseeding, and the mean is
    taken over batches rather than samples, so the two agree.
    """
    losses: list[float] = []

    was_training = policy.training
    policy.eval()
    try:
        # Wraps the loader too: iterating a DataLoader consumes the main process's torch RNG for its
        # worker seeds, which would otherwise shift the training stream by one draw per evaluation.
        with frozen_rng(seed), accelerator.autocast():
            for batch_idx, batch in enumerate(loader):
                batch = preprocessor(batch)
                # Identical noise and time for this batch at every checkpoint.
                torch.manual_seed(seed + batch_idx)
                loss, _ = policy.forward(batch)
                losses.append(loss.item())
    finally:
        policy.train(was_training)

    if not losses:
        return float("nan"), float("nan"), 0

    mean = sum(losses) / len(losses)
    # NaN rather than 0.0 for a single batch: zero would read as "no variance" when it means
    # "not estimated".
    if len(losses) > 1:
        std = (sum((x - mean) ** 2 for x in losses) / (len(losses) - 1)) ** 0.5
    else:
        std = float("nan")

    return mean, std, len(loader.dataset)  # type: ignore[arg-type]


def append_offline_eval_row(output_dir, step: int, mean: float, std: float, n_samples: int) -> None:
    """Append one evaluation to `offline_eval.csv` in the run directory.

    Columns match `lerobot_validate_offline`'s CSV so the in-training curve and a later sweep over the
    saved checkpoints can be concatenated into one plot.
    """
    path = Path(output_dir) / "offline_eval.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "mean_loss", "std_loss", "n_samples"])
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "checkpoint": f"{step:06d}",
                "step": step,
                "mean_loss": mean,
                "std_loss": std,
                "n_samples": n_samples,
            }
        )
