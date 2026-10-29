#!/usr/bin/env python

"""Processor steps for converting between absolute and relative actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from src.lerobot.configs.types import PipelineFeatureType, PolicyFeature
from src.lerobot.processor.core import EnvTransition, TransitionKey
from src.lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

__all__ = [
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def _apply_state_offset(actions: Tensor, state: Tensor, mask: Sequence[bool], sign: int) -> Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = min(mask_tensor.numel(), actions.shape[-1], state.shape[-1])
    if dims == 0:
        return actions.clone()
    state = state.to(device=actions.device, dtype=actions.dtype)
    offset = state[..., :dims] * mask_tensor[:dims]
    if actions.ndim == 3:
        offset = offset.unsqueeze(-2)
    result = actions.clone()
    result[..., :dims] += sign * offset
    return result


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to offsets from the current state."""
    return _apply_state_offset(actions, state, mask, sign=-1)


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert action offsets back to absolute actions."""
    return _apply_state_offset(actions, state, mask, sign=1)


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Convert selected action dimensions to offsets relative to observation state."""

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim
        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        mask = [
            not any(token == str(name).lower() or token in str(name).lower() for token in exclude_tokens)
            for name in self.action_names[:action_dim]
        ]
        return mask + [True] * max(0, action_dim - len(mask))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        state = observation.get(OBS_STATE)
        if state is not None:
            self._last_state = state
        result = transition.copy()
        if not self.enabled:
            return result
        actions = result.get(TransitionKey.ACTION)
        if actions is None or state is None:
            return result
        if not isinstance(actions, Tensor):
            actions = torch.as_tensor(actions)
        if not isinstance(state, Tensor):
            state = torch.as_tensor(state)
        result[TransitionKey.ACTION] = to_relative_actions(actions, state, self._build_mask(actions.shape[-1]))
        return result

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Restore predicted relative actions to absolute actions."""

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("AbsoluteActionsProcessorStep requires a paired relative action processor")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("No state was cached by the relative action processor")
        result = transition.copy()
        actions = result.get(TransitionKey.ACTION)
        if actions is None:
            return result
        if not isinstance(actions, Tensor):
            actions = torch.as_tensor(actions)
        result[TransitionKey.ACTION] = to_absolute_actions(
            actions, state, self.relative_step._build_mask(actions.shape[-1])
        )
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
