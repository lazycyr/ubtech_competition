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
"""Held-out validation loss for trained checkpoints, without an environment.

Why this exists
---------------
`lerobot_train.py` only evaluates through a simulator (`cfg.env`), so an offline
training run produces no validation signal at all. Training loss cannot stand in
for one: for flow-matching policies it keeps decreasing towards ~0 as the model
memorises the training set, and plateaus never appear. Picking a checkpoint from
training loss alone is guesswork.

This script computes the same flow-matching loss on episodes that were *not* part
of training, so the curve has a minimum you can actually pick.

Two details make the numbers trustworthy:

*   **The same noise and time draws are used for every checkpoint.** pi05 samples
    both per forward pass (`modeling_pi05.py:358-359`), so the raw loss is
    stochastic. We reseed with `seed + batch_index` before each forward, which
    makes the sample identical across checkpoints -- otherwise the draw-to-draw
    variance swamps the difference you are trying to measure.
*   **Held-out episodes must be disjoint from training.** Datasets under
    `datasets/` are prefixes of each other (`Part_Sorting_300/selected` is
    `Part_Sorting[:300]`, verified by hashing episode action data), so only the
    tail of the parent set is genuinely unseen. See `--val_episodes`.

There is no memory cost relative to training (no gradients, no optimizer state),
so `--batch_size` can be much larger than the training one.

Example
-------
    sudo /isaac-sim/python.sh -m src.lerobot.scripts.lerobot_validate_offline \
        --checkpoints_dir=outputs/checkpoints/pi05/Part_Sorting_300_full/checkpoints \
        --dataset_root=datasets/Part_Sorting \
        --dataset_repo_id=cyr/all \
        --val_episodes=300-2999 \
        --max_samples=4096 \
        --batch_size=8 \
        --output_file=outputs/val/Part_Sorting_300_full.csv

CPU validation (useful while the GPUs are busy with training) needs
`CUDA_VISIBLE_DEVICES=""` in the environment as well as `--device=cpu`, and
`--max_samples` in the low hundreds at most -- roughly 5-8 minutes per
checkpoint, dominated by loading the weights.

LoRA / PEFT checkpoints are supported. The loading path is chosen from the
checkpoint directory's contents rather than from `train_config.json`, which
records `use_peft: false` even for LoRA runs.
"""

import copy
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
from torch.utils.data import DataLoader, Subset

from src.lerobot.configs.train import TrainPipelineConfig
from src.lerobot.datasets.factory import make_dataset
from src.lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from src.lerobot.policies.factory import make_policy, make_pre_post_processors
from src.lerobot.utils.import_utils import register_third_party_plugins

TRAIN_CONFIG_NAME = "train_config.json"
PRETRAINED_MODEL_DIR = "pretrained_model"


@dataclass
class OfflineValidationConfig:
    """Configuration for `lerobot_validate_offline`."""

    # Directory holding checkpoints to score. Either a run's `checkpoints/` dir
    # (every `<step>/pretrained_model/` inside it is evaluated, ordered by step)
    # or a single `pretrained_model/` directory.
    checkpoints_dir: str = ""

    # Held-out data. Point `dataset_root` at the *parent* dataset and select the
    # unseen episode range with `val_episodes`.
    dataset_root: str = ""
    dataset_repo_id: str = ""

    # Which episodes to score. Accepts "300-2999" (inclusive), "300:3000"
    # (half-open) or an explicit list "1,5,9". Empty means every episode in the
    # dataset -- only safe if the dataset was never trained on.
    val_episodes: str = ""

    # Frames drawn from those episodes. One sample is produced per frame, so a
    # 300-episode set is ~190k samples; subsampling keeps a pass to seconds while
    # staying well above the noise floor of the loss estimate.
    max_samples: int = 4096

    # No gradients or optimizer state are held, so this can be much larger than
    # the training batch size.
    batch_size: int = 8
    num_workers: int = 4

    # Fixes both the frame subsample and the noise/time draws. Keep it constant
    # across checkpoints or the comparison is meaningless.
    seed: int = 0

    device: str = "cuda"

    # Optional CSV output: checkpoint, step, mean_loss, std_loss, n_samples.
    output_file: str | None = None

    # Stop after this many batches. Only useful to smoke-test the plumbing.
    limit_batches: int | None = None


def parse_episodes(spec: str, available: list[int]) -> list[int]:
    """Expand "300-2999" / "300:3000" / "1,5,9" into a list of episode indices."""
    spec = spec.strip()
    if not spec:
        return sorted(available)

    if "," in spec:
        selected = [int(x) for x in spec.split(",") if x.strip()]
    else:
        m = re.fullmatch(r"(\d+)\s*([-:])\s*(\d+)", spec)
        if not m:
            raise ValueError(
                f"Cannot parse --val_episodes={spec!r}. Use '300-2999', '300:3000' or '1,5,9'."
            )
        start, end = int(m.group(1)), int(m.group(3))
        # "-" is inclusive on both ends; ":" is half-open, matching Python
        # slicing habits while keeping the obvious "300-2999 means 2700" reading.
        selected = list(range(start, end + 1 if m.group(2) == "-" else end))

    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"--val_episodes selects {len(missing)} episode(s) not present in the dataset, "
            f"e.g. {missing[:5]}. Available range: {min(available)}..{max(available)}"
        )
    return selected


def is_checkpoint_dir(path: Path) -> bool:
    """Whether `path` holds a loadable policy.

    Full fine-tunes store `model.safetensors`; LoRA runs store
    `adapter_model.safetensors` instead. Requiring the former would silently
    skip every PEFT checkpoint.
    """
    if not (path / "config.json").exists():
        return False
    return (path / "model.safetensors").exists() or (path / "adapter_model.safetensors").exists()


def discover_checkpoints(root: str) -> list[tuple[int, Path]]:
    """Return [(step, pretrained_model_dir)] sorted by step.

    Handles both a run-level `checkpoints/` directory and a single
    `pretrained_model/` directory. Symlinks such as the `last` pointer are
    resolved and de-duplicated.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"--checkpoints_dir does not exist: {root_path}")

    # A single checkpoint directory was passed directly.
    if is_checkpoint_dir(root_path):
        return [(0, root_path)]

    found: dict[Path, int] = {}
    for child in sorted(root_path.iterdir()):
        candidate = child / PRETRAINED_MODEL_DIR if (child / PRETRAINED_MODEL_DIR).is_dir() else child
        if not is_checkpoint_dir(candidate):
            continue

        # resolve() collapses the `last -> 001000` symlink so it is not scored twice
        resolved = candidate.resolve()
        m = re.fullmatch(r"0*(\d+)", child.name)
        found.setdefault(resolved, int(m.group(1)) if m else 0)

    if not found:
        raise FileNotFoundError(f"No `{PRETRAINED_MODEL_DIR}/` directories found under {root_path}")

    return sorted(((step, path) for path, step in found.items()), key=lambda t: t[0])


def build_validation_dataset(cfg: OfflineValidationConfig, first_ckpt: Path):
    """Build the held-out dataset, reusing the training config of the run.

    Reusing `TrainPipelineConfig.from_pretrained` on the checkpoint's
    `train_config.json` keeps `chunk_size`, `n_action_steps` and the delta
    timestamps identical to training -- otherwise the loss would be computed
    against a differently-shaped action horizon and be incomparable.
    """
    train_config_path = first_ckpt / TRAIN_CONFIG_NAME
    if not train_config_path.exists():
        raise FileNotFoundError(
            f"{train_config_path} not found. The checkpoint must come from lerobot_train.py, "
            "which writes train_config.json next to the weights."
        )

    tcfg = TrainPipelineConfig.from_pretrained(train_config_path)

    if cfg.dataset_root:
        tcfg.dataset.root = cfg.dataset_root
    if cfg.dataset_repo_id:
        tcfg.dataset.repo_id = cfg.dataset_repo_id
    # No augmentation during validation -- it would add variance for no benefit.
    tcfg.dataset.image_transforms.enable = False

    if cfg.val_episodes:
        # Resolve against the parent dataset's real episode index list rather
        # than assuming a contiguous 0..N-1 range: a mistyped range then fails
        # loudly here instead of silently scoring a different set than intended.
        meta = LeRobotDatasetMetadata(tcfg.dataset.repo_id, root=tcfg.dataset.root)
        tcfg.dataset.episodes = parse_episodes(cfg.val_episodes, list(meta.episodes["episode_index"]))
    else:
        tcfg.dataset.episodes = None

    return tcfg, make_dataset(tcfg)


def subsample_indices(n_frames: int, max_samples: int, seed: int) -> list[int]:
    """Deterministically pick `max_samples` frame indices out of `n_frames`."""
    if max_samples >= n_frames:
        return list(range(n_frames))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_frames, generator=generator)[:max_samples].sort().values.tolist()


def _batch_size_of(batch: dict) -> int:
    """Leading dimension of the first tensor in a batch dict.

    Preprocessed batches can carry non-tensor entries (task strings, metadata),
    so `next(iter(...))` is not safe.
    """
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("No tensor with a batch dimension found in the batch")


@torch.no_grad()
def evaluate_checkpoint(
    cfg: OfflineValidationConfig,
    tcfg: TrainPipelineConfig,
    ckpt_dir: Path,
    dataset,
    loader: DataLoader,
) -> tuple[float, float, int]:
    """Mean / std flow-matching loss of one checkpoint over the held-out set."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Own copy: `make_policy` reads these and we must not leak the checkpoint
    # path into the next iteration.
    policy_cfg = copy.deepcopy(tcfg.policy)
    policy_cfg.pretrained_path = ckpt_dir
    policy_cfg.device = device.type

    # `make_policy` picks its loading path off this flag (factory.py:539/555):
    # False loads `pretrained_path` as a full model, True parses it as a PEFT
    # adapter and takes the base model from the adapter's own config.
    # `train_config.json` cannot be trusted here -- it records `use_peft: false`
    # even for LoRA runs, because what enabled LoRA lives at train level
    # (`cfg.peft`, see lerobot_train.py:249). Trust the directory contents.
    policy_cfg.use_peft = (ckpt_dir / "adapter_config.json").exists()

    if policy_cfg.use_peft:
        # The base model the adapter was trained on is recorded here and loaded
        # by `make_policy`. Relative paths resolve against the *current working
        # directory*, so running from anywhere but the directory the run was
        # launched from fails with a confusing HF repo-id error instead.
        adapter_cfg = json.loads((ckpt_dir / "adapter_config.json").read_text())
        base = adapter_cfg.get("base_model_name_or_path")
        if base and not Path(base).is_absolute() and not Path(base).exists():
            logging.warning(
                f"    adapter base model {base!r} not found relative to {Path.cwd()} -- "
                "run this script from the directory the checkpoint was saved in (usually the repo root)."
            )

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=tcfg.rename_map)
    policy.to(device)
    policy.eval()

    # The saved preprocessor carries the training set's normalisation statistics,
    # so the loss is computed in exactly the space training optimised in. Only
    # the device has to be overridden (training did the same).
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    autocast_ctx = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and "bfloat16" in str(policy_cfg.dtype),
    )

    losses: list[float] = []
    n_samples = 0
    for batch_idx, batch in enumerate(loader):
        if cfg.limit_batches is not None and batch_idx >= cfg.limit_batches:
            break

        size = _batch_size_of(batch)
        batch = preprocessor(batch)

        # Reseeding per batch index is what makes checkpoints comparable: every
        # checkpoint sees the same noise/time for a given batch, so differences
        # in the mean reflect the weights rather than the sampling.
        torch.manual_seed(cfg.seed + batch_idx)
        with autocast_ctx:
            loss, _ = policy.forward(batch)

        losses.append(loss.item())
        n_samples += size

    del policy, preprocessor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not losses:
        return float("nan"), float("nan"), 0

    mean = sum(losses) / len(losses)
    # NaN rather than 0.0 for a single batch: zero would read as "no variance"
    # when it actually means "not estimated".
    if len(losses) > 1:
        std = (sum((x - mean) ** 2 for x in losses) / (len(losses) - 1)) ** 0.5
    else:
        std = float("nan")
    return mean, std, n_samples


@draccus.wrap()
def validate_offline(cfg: OfflineValidationConfig) -> None:
    register_third_party_plugins()
    # force=True is required: importing the lerobot modules installs a root
    # handler first, and plain basicConfig() is a no-op once one exists. Without
    # it every logging.info below is silently dropped.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    if cfg.device == "cpu" and torch.cuda.is_available():
        logging.warning(
            "--device=cpu but CUDA is visible. For LoRA checkpoints this is not enough: PEFT picks its "
            "load device with `infer_device()` (peft/peft_model.py:1461), which returns cuda whenever it "
            "is available, ignoring the model's own device. Loading an adapter on a GPU already held by "
            "training will OOM. Relaunch with CUDA_VISIBLE_DEVICES=\"\" to force CPU."
        )

    checkpoints = discover_checkpoints(cfg.checkpoints_dir)
    logging.info(f"Found {len(checkpoints)} checkpoint(s) under {cfg.checkpoints_dir}")

    tcfg, dataset = build_validation_dataset(cfg, checkpoints[0][1])
    logging.info(
        f"Held-out set: {len(dataset)} frames from {len(dataset.episodes)} episodes "
        f"({cfg.dataset_root or tcfg.dataset.root}, episodes {cfg.val_episodes or 'all'})"
    )

    indices = subsample_indices(len(dataset), cfg.max_samples, cfg.seed)
    split = Subset(dataset, indices)

    # shuffle=False keeps the batch_idx -> sample mapping stable across
    # checkpoints, which the per-batch reseeding above depends on.
    loader = DataLoader(
        split,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    logging.info(
        f"Scoring {len(indices)} of {len(dataset)} frames "
        f"({len(loader)} batches of {cfg.batch_size}, seed={cfg.seed})"
    )

    rows: list[dict] = []
    for step, ckpt_dir in checkpoints:
        label = ckpt_dir.parent.name if ckpt_dir.name == PRETRAINED_MODEL_DIR else ckpt_dir.name
        logging.info(f"--- {label} (step {step}) ---")
        mean, std, n = evaluate_checkpoint(cfg, tcfg, ckpt_dir, dataset, loader)
        rows.append({"checkpoint": label, "step": step, "mean_loss": mean, "std_loss": std, "n_samples": n})
        logging.info(f"    mean loss {mean:.6f}  (std {std:.6f} across batches, n={n})")

    print()
    best = min(rows, key=lambda r: r["mean_loss"])
    print(f"{'checkpoint':>12} {'step':>8} {'mean_loss':>12} {'std':>10} {'n':>8}")
    for r in rows:
        marker = "  <-- lowest" if r is best else ""
        print(
            f"{r['checkpoint']:>12} {r['step']:>8} {r['mean_loss']:>12.6f} "
            f"{r['std_loss']:>10.6f} {r['n_samples']:>8}{marker}"
        )
    print()

    if cfg.output_file:
        out = Path(cfg.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "mean_loss", "std_loss", "n_samples"])
            writer.writeheader()
            writer.writerows(rows)
        # The noise/time seed is part of what the numbers mean, not just metadata.
        meta_path = out.with_suffix(out.suffix + ".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "checkpoints_dir": cfg.checkpoints_dir,
                    "dataset_root": cfg.dataset_root,
                    "dataset_repo_id": cfg.dataset_repo_id,
                    "val_episodes": cfg.val_episodes,
                    "max_samples": cfg.max_samples,
                    "batch_size": cfg.batch_size,
                    "seed": cfg.seed,
                },
                indent=2,
            )
        )
        logging.info(f"Wrote {out} and {meta_path}")


if __name__ == "__main__":
    validate_offline()
