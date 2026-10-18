"""KITTICarlaAdapter — simulated source domain, NO real radar.

`has_real_radar()` returns False, which is not a detail: it means association,
soft labels, head fitting, calibration and the swap test are all unavailable
here, and every fitted artifact must be imported from a donor dataset
(`fitted_from: vod`). That asymmetry is the sim2real problem stated in code.

Note also that KITTI-CARLA has no separate radar mount -- the virtual radar
sits at the LiDAR origin and the instance-derived boxes share that frame -- so
`to_pcdet_frame` is the identity here while VoD needs inv(radar_to_lidar).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rear.data.base import DatasetAdapter, Frame


class KITTICarlaAdapter(DatasetAdapter):
    def __init__(self, config, lazy=False, root=None, towns=None, split=None):
        """`split` is accepted for interface uniformity with adapters that have
        train/val/test splits, and selects a TOWN here. KITTI-CARLA's sequences
        are towns and it has no held-out real radar to score against, so a
        train/val/test division of it would be decorative -- the honest thing is
        to expose the towns and let the caller choose."""
        super().__init__(config)
        self.split = split
        self.root = Path(root or config["root"])
        self.towns = list(towns or config.get("towns", ["Town01", "Town02"]))
        if split is not None and split in self.towns:
            self.towns = [split]
        self._ds = None
        if not lazy:
            self._open()

    def _open(self):
        if self._ds is None:
            from rear.data._kitti_carla_loader import KittiCarlaDataset
            self._ds = [KittiCarlaDataset(str(self.root / t), load_rgb=False)
                        for t in self.towns if (self.root / t).exists()]
        return self._ds

    def has_real_radar(self) -> bool:
        return False

    def __len__(self) -> int:
        return sum(len(d) for d in self._open())

    def _locate(self, idx):
        off = 0
        for t, d in enumerate(self._open()):
            if idx - off < len(d):
                return t, int(idx - off)
            off += len(d)
        raise IndexError(idx)

    def load_frame(self, idx) -> Frame:
        """`idx` is either a flat int (0-based across all open towns, in
        `self.towns` order) or a `"Town:NNNN"` frame_id -- the form
        `sequence_splits()` emits, so callers that iterate a split's ids
        directly (as `scripts/generate.py` does) don't need to re-derive a
        flat index themselves."""
        if isinstance(idx, str):
            town, j_str = idx.split(":")
            t, j = self.towns.index(town), int(j_str)
        else:
            t, j = self._locate(int(idx))
        f = self._open()[t][j]
        flat = sum(len(self._open()[k]) for k in range(t)) + j
        return Frame(index=flat, frame_id=f"{self.towns[t]}:{j:04d}",
                     lidar=f.lidar, ego_velocity=f.ego_velocity,
                     calibration={"radar_to_lidar": np.eye(4)},
                     boxes=[], radar=None, camera=getattr(f, "image", None),
                     lidar_semantic=f.lidar_semantic,
                     lidar_instance=f.lidar_instance)

    def sequence_splits(self) -> dict:
        """Towns ARE the sequences here; there is no held-out real radar to
        evaluate against, so this exists for provenance, not for scoring."""
        out, off = {}, 0
        for t, d in zip(self.towns, self._open()):
            out[t] = [f"{t}:{j:04d}" for j in range(len(d))]
            off += len(d)
        return out

    def boxes_for(self, idx: int):
        """Oriented boxes derived from CARLA instance+semantic tags."""
        from rear.data._kc_boxes import boxes_from_instances
        f = self.load_frame(idx)
        return boxes_from_instances(f.lidar[:, :3], f.lidar_semantic, f.lidar_instance)
