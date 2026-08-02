from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from rsl_rl.models.mlp_model import MLPModel

from .vjepa_lite import LightVJEPAEncoder


class VJEPAFusionModel(MLPModel):
    """Actor/critic model fusing proprioceptive obs (velocity + lidar) with a light V-JEPA-style visual
    latent extracted from the robot's forward camera.

    Mirrors rsl_rl's own ``CNNModel`` pattern: the vision encoder is sized/constructed before calling
    ``super().__init__()`` (its latent dim needs to be known to size the trunk MLP) but only assigned to
    ``self`` afterwards, since ``nn.Module`` bookkeeping isn't set up until ``nn.Module.__init__`` runs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        vjepa_cfg: dict | None = None,
    ) -> None:
        self._get_obs_dim(obs, obs_groups, obs_set)
        vjepa_encoder = LightVJEPAEncoder(**(vjepa_cfg or {}))
        self._vjepa_latent_dim = vjepa_encoder.latent_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.vjepa = vjepa_encoder

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Split active observation groups into 1D proprioceptive groups and the single 2D image group."""
        active_obs_groups = obs_groups[obs_set]
        vector_groups: list[str] = []
        vector_dim = 0
        image_groups: list[str] = []
        for obs_group in active_obs_groups:
            if obs[obs_group].dim() == 4:
                image_groups.append(obs_group)
            else:
                vector_groups.append(obs_group)
                vector_dim += obs[obs_group].shape[-1]
        if len(image_groups) != 1:
            raise ValueError(f"VJEPAFusionModel expects exactly one image observation group, got {image_groups}.")
        self._image_group = image_groups[0]
        return vector_groups, vector_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._vjepa_latent_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        """Concatenate the normalized proprioceptive latent with the V-JEPA visual latent."""
        latent_vector = super().get_latent(obs)
        latent_image = self.vjepa(obs[self._image_group])
        return torch.cat([latent_vector, latent_image], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchVJEPAModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxVJEPAModel(self, verbose)


class _TorchVJEPAModel(nn.Module):
    """Exportable VJEPAFusionModel for JIT.

    MLPModel's own ``as_jit()`` assumes a single flat vector input, so it can't be reused here: this
    model also consumes a raw image tensor that must go through the (copied) V-JEPA encoder before being
    concatenated with the normalized vector latent, exactly mirroring rsl_rl's own ``_TorchCNNModel``.
    """

    def __init__(self, model: VJEPAFusionModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.vjepa = copy.deepcopy(model.vjepa)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_vector: torch.Tensor, obs_image: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from separated vector and image inputs."""
        latent_vector = self.obs_normalizer(obs_vector)
        latent_image = self.vjepa(obs_image)
        latent = torch.cat([latent_vector, latent_image], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for this model)."""
        pass


class _OnnxVJEPAModel(nn.Module):
    """Exportable VJEPAFusionModel for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: VJEPAFusionModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.vjepa = copy.deepcopy(model.vjepa)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.vector_dim = model.obs_dim
        self.image_size = model.vjepa.image_size
        self.image_channels = model.vjepa.in_channels

    def forward(self, obs_vector: torch.Tensor, obs_image: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_vector = self.obs_normalizer(obs_vector)
        latent_image = self.vjepa(obs_image)
        latent = torch.cat([latent_vector, latent_image], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.vector_dim),
            torch.zeros(1, self.image_size, self.image_size, self.image_channels),
        )

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs_vector", "obs_image"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


@configclass
class RslRlVJEPAModelCfg(RslRlMLPModelCfg):
    """Model configuration wiring :class:`VJEPAFusionModel` into rsl_rl via its ``class_name`` resolver."""

    class_name: str = "isaaclab_tasks.manager_based.turtlebot3.models:VJEPAFusionModel"

    @configclass
    class VJEPACfg:
        """Constructor arguments for :class:`~isaaclab_tasks.manager_based.turtlebot3.vjepa_lite.LightVJEPAEncoder`."""

        image_size: int = 64
        patch_size: int = 8
        embed_dim: int = 96
        depth: int = 4
        num_heads: int = 3
        latent_dim: int = 128

    vjepa_cfg: VJEPACfg = VJEPACfg()