"""The DatasetAdapter abstraction — the piece that makes this framework portable.

Before this refactor, every dataset-specific fact lived somewhere in code: the
radar mount offset was a literal in three scripts, the 60 deg FOV was a module
constant named `VOD_RADAR_FOV_DEG`, the voxel size was a dataclass default, and
the fitted Platt/RCS/clutter parameters were pickles found by hardcoded path.
Adding a dataset meant editing all of it.

Now a dataset is two things and nothing else:

  1. an adapter -- how to READ it (frames, splits, sensor geometry)
  2. a YAML descriptor -- what was FITTED on it (configs/datasets/<name>.yaml)

Everything in `rear/` reads both through this interface. No module under
`rear/` may contain a dataset-specific number; `tests/` enforces that by
grepping for the known offenders.

WHAT A NEW DATASET MUST PROVIDE
-------------------------------
Minimum, to GENERATE radar: LiDAR points, ego velocity, per-frame sensor
calibration, sequence-level splits.
Additionally, to FIT on it (scripts/fit_dataset.py): real paired radar.
Additionally, to run the swap test: a camera.

`has_real_radar()` is the switch that decides which of those a dataset can
support. KITTI-CARLA returns False, which is exactly why it imports every
fitted artifact from VoD -- the asymmetry the whole project is about.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def _expand(v):
    """Resolve ${VAR:default} in config strings so roots are machine-portable."""
    if isinstance(v, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_dataset_config(name_or_path) -> dict:
    """Load configs/datasets/<name>.yaml (or an explicit path)."""
    p = Path(name_or_path)
    if not p.exists():
        p = REPO_ROOT / "configs" / "datasets" / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no dataset config for {name_or_path!r} ({p})")
    return _expand(yaml.safe_load(p.read_text()))


@dataclass
class Frame:
    """One synchronised multi-sensor frame. `radar` is None where the dataset
    has no real radar -- callers must check rather than assume."""
    index: int
    frame_id: str
    lidar: np.ndarray                     # (N, 4) x, y, z, <intensity|cosine>
    ego_velocity: np.ndarray              # (3,) m/s, LiDAR-frame axes
    calibration: dict                     # at least {'radar_to_lidar': 4x4}
    boxes: list = field(default_factory=list)
    radar: np.ndarray | None = None       # (M, C) or None
    camera: np.ndarray | None = None      # (H, W, 3) uint8 or None
    lidar_semantic: np.ndarray | None = None
    lidar_instance: np.ndarray | None = None


class DatasetAdapter(ABC):
    """Read-side interface. Implement these six things and the whole framework
    -- generation, validation, null controls, swap test, downstream -- works."""

    def __init__(self, config: dict):
        self.config = config
        self.name = config["name"]
        s = config["sensors"]
        self.radar_mount_offset = np.asarray(s["radar_mount_offset"], dtype=np.float64)
        self.radar_fov_degrees = float(s["radar_fov_degrees"])
        self.lidar_sensor_type = s["lidar_sensor_type"]
        self.pcdet_frame = s.get("pcdet_frame", "lidar")
        self.generator_frame = s.get("generator_frame", "lidar")

    # ---- required -------------------------------------------------------
    @abstractmethod
    def load_frame(self, idx: int) -> Frame: ...

    @abstractmethod
    def sequence_splits(self) -> dict:
        """{'train': [frame_id, ...], 'val': [...], 'test': [...]}, split by
        SEQUENCE. Frame-level splits are forbidden (CLAUDE.md): consecutive
        frames are ~0.1 s apart and near-duplicates."""

    @abstractmethod
    def has_real_radar(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # ---- provided -------------------------------------------------------
    def has_camera(self) -> bool:
        return bool(self.config["sensors"].get("has_camera", False))

    def fitted(self, *keys, default=None) -> Any:
        """Read a fitted artifact value. Follows `fitted_from` so a dataset with
        no real radar (KITTI-CARLA) transparently inherits the donor's fit."""
        node = self.config.get("fitted", {})
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                donor = self.config.get("fitted_from")
                if donor:
                    return load_adapter_config(donor).fitted(*keys, default=default)
                return default
            node = node[k]
        return node

    def artifact(self, *keys) -> Path:
        """Absolute path to a fitted artifact named in the config."""
        rel = self.fitted(*keys)
        if rel is None:
            raise KeyError(f"no artifact at {keys} for dataset {self.name}")
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    def grid_config(self, radar_position=None):
        """A VoxelGridConfig built ENTIRELY from this dataset's config."""
        from rear.geometry.candidates import VoxelGridConfig
        c = self.fitted("candidates") or {}
        return VoxelGridConfig(
            voxel_size=c.get("voxel_size", 0.4),
            x_range=tuple(c.get("x_range", (-50.0, 50.0))),
            y_range=tuple(c.get("y_range", (-50.0, 50.0))),
            z_range=tuple(c.get("z_range", (-3.0, 5.0))),
            radar_position=tuple(radar_position if radar_position is not None
                                 else self.radar_mount_offset),
            radar_fov_deg=self.radar_fov_degrees,
            max_occlusion_depth=c.get("max_occlusion_depth", 3),
            min_consecutive_blocking_voxels=c.get("min_consecutive_blocking_voxels", 1),
            apply_fov_mask=c.get("apply_fov_mask", True),
            cos_angle_source=c.get("cos_angle_source", "column3"),
            normal_radius=c.get("normal_radius", 0.6),
            normal_k_neighbors=c.get("normal_k_neighbors", 10),
            ground_z_source=c.get("ground_z_source", "geometric_percentile"),
            voxel_size_schedule=c.get("voxel_size_schedule", "fixed"),
        )

    def material_config(self) -> dict:
        """This dataset's material-information settings, `fitted_from`-aware.

        Material is a PERMANENT component of the pipeline, so this returns an
        enabled block for any dataset that declares one and an explicitly
        disabled block otherwise -- never a guess. The two keys that make it
        dataset-independent:

          label_space  which file under configs/materials/ maps this dataset's
                       semantic ids to material mixtures. A new taxonomy is a
                       new file there, not a code change.
          source       `lidar_semantic` if the dataset ships per-point labels,
                       `camera_segmentation` if they must be predicted from the
                       camera and projected (see rear/materials/pipeline.py).

        `frequency_ghz` and the segmentation model are read here too, so nothing
        about the material path is hardcoded in `rear/`.
        """
        node = self.config.get("materials")
        if node is None:
            donor = self.config.get("fitted_from")
            if donor:
                return load_adapter_config(donor).material_config()
            return {"enabled": False}
        out = dict(node)
        out.setdefault("enabled", True)
        if out["enabled"] and "label_space" not in out:
            raise ValueError(
                f"{self.name}: `materials:` is enabled but names no "
                f"`label_space`. Point it at a file in configs/materials/.")
        return out

    def camera_projection(self, frame: Frame):
        """(P, R0, Tr) mapping LiDAR-frame points into this frame's camera image.

        Required only by datasets whose material source is
        `camera_segmentation`; a dataset shipping per-point semantics never
        calls it. Raises rather than returning an identity guess, because a
        wrong projection would silently attach every material label to the wrong
        geometry.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement camera_projection(). It "
            f"is needed because this dataset's materials.source is "
            f"'camera_segmentation'; return (P, R0_rect, Tr_velo_to_cam) from "
            f"its own calibration.")

    def to_pcdet_frame(self, xyz_lidar: np.ndarray, calibration: dict) -> np.ndarray:
        """Map generator output into whatever frame this dataset's pcdet trees
        and labels use.

        This is the single place the Phase-1 coordinate bug can live, so it
        lives here once instead of in every build script. VoD's trees are in the
        NATIVE RADAR frame and need inv(radar_to_lidar); KITTI-CARLA's are
        already in the LiDAR frame and need nothing.
        """
        if self.pcdet_frame == self.generator_frame:
            return np.asarray(xyz_lidar, dtype=np.float64)
        T = np.asarray(calibration["radar_to_lidar"], dtype=np.float64)
        h = np.concatenate([np.asarray(xyz_lidar, dtype=np.float64),
                            np.ones((len(xyz_lidar), 1))], axis=1)
        return (np.linalg.inv(T) @ h.T).T[:, :3]

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, frames={len(self)}, "
                f"real_radar={self.has_real_radar()}, camera={self.has_camera()})")


def load_adapter(name_or_path, **kw) -> DatasetAdapter:
    """Instantiate the adapter named by a dataset config's `adapter:` field."""
    cfg = load_dataset_config(name_or_path)
    return _instantiate(cfg, **kw)


def load_adapter_config(name_or_path) -> DatasetAdapter:
    """Adapter built lazily for config lookups only (no data access)."""
    return _instantiate(load_dataset_config(name_or_path), lazy=True)


def _instantiate(cfg, lazy=False, **kw):
    import importlib
    mod_name, _, cls_name = cfg["adapter"].partition(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(cfg, lazy=lazy, **kw)
