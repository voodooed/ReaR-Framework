"""VoDAdapter — View-of-Delft. The only dataset here with real paired radar.

NOT NEEDED to generate radar from LiDAR (that only needs `configs/datasets/
kitti_carla.yaml` + `weights/`). This adapter exists because `kitti_carla.yaml`
declares `fitted_from: vod`, so `configs/datasets/vod.yaml` is read at
generation time to resolve inherited parameters -- but that only touches the
YAML, never this class's `_open()`.

`_open()` (real frame loading, splits, ego-velocity repair) requires a
`data.vod_loader.VoDDataset` implementation that is NOT vendored in this
package -- VoD is a licensed dataset with its own loader/scene-index code that
lived in the original research repo. Bring your own `vod_loader.py` on
`sys.path` (matching the interface used below) if you need to generate radar
FOR VoD LiDAR or re-fit these weights on a new dataset; see USAGE.md.
"""

from __future__ import annotations

import numpy as np

from rear.data.base import DatasetAdapter, Frame


class VoDAdapter(DatasetAdapter):
    def __init__(self, config, lazy=False, split="train", root=None):
        super().__init__(config)
        self.split = split
        self.root = str(root or config["root"])
        self._ds_by_split: dict = {}
        if not lazy:
            self._open()

    def _open(self, split=None):
        """Lazily open (and cache) the VoDDataset for one split. Defaults to
        this adapter's own `self.split`, so a plain positional int index
        keeps meaning what it always did: a position within the split this
        adapter was constructed with. A frame_id STRING is resolved to
        whichever split actually contains it (see `_locate`), opening that
        split's dataset on demand -- so one adapter instance can serve
        train/val/test frame ids interchangeably, exactly what
        DatasetAdapter.sequence_splits() promises callers that iterate its
        emitted ids directly (scripts/generate.py, scripts/fit_dataset.py)."""
        split = split or self.split
        if split not in self._ds_by_split:
            from data.vod_loader import VoDDataset
            self._ds_by_split[split] = VoDDataset(self.root, split=split)
        return self._ds_by_split[split]

    def has_real_radar(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._open().frame_ids)

    def _locate(self, frame_id: str):
        """(split, position) of a frame_id string. Checks this adapter's own
        split first -- the common case -- before the other two."""
        order = [self.split] + [s for s in ("train", "val", "test") if s != self.split]
        for s in order:
            ds = self._open(s)
            if frame_id in ds.frame_ids:
                return s, ds.frame_ids.index(frame_id)
        raise KeyError(f"frame_id {frame_id!r} not found in any of train/val/test")

    def load_frame(self, idx) -> Frame:
        """`idx` is either a plain positional int (a position within THIS
        adapter's own `self.split`, unchanged pre-existing behaviour) or a
        frame_id STRING as `sequence_splits()` emits -- resolved to whichever
        split actually contains it, so a caller iterating a split's id list
        directly does not need to know or re-derive which split each id
        belongs to (mirrors KITTICarlaAdapter's `"Town:NNNN"` string ids)."""
        if isinstance(idx, str):
            split, pos = self._locate(idx)
            f = self._open(split)[pos]
            index = -1  # no single flat index is meaningful across splits
        else:
            f = self._open()[int(idx)]
            index = int(idx)
        return Frame(index=index, frame_id=str(f.frame_id), lidar=f.lidar,
                     ego_velocity=f.ego_velocity, calibration=f.calibration,
                     boxes=f.boxes, radar=f.radar, camera=f.image)

    def sequence_splits(self) -> dict:
        return {s: list(self._open(s).frame_ids) for s in ("train", "val", "test")}

    def frame_ids(self):
        return list(self._open().frame_ids)

    def sequence_of(self) -> dict:
        """{frame_id: sequence index} -- needed by every sequence-level split
        and by the null controls that must not straddle a sequence."""
        ds = self._open()
        out = {}
        for si, tok in enumerate(ds._scene_index.scene_tokens_by_first_frame()):
            for fid in ds._scene_index.frames_in_scene(tok):
                out[str(fid)] = si
        return out

    def camera_projection(self, frame: Frame):
        """(P2, R0_rect, Tr_velo_to_cam) for this frame, from VoD's own calib
        file -- the projection the material pipeline transfers camera labels
        onto LiDAR points with.

        Parsed per frame from `lidar/training/calib/<id>.txt` rather than
        rebuilt from the `lidar_to_camera` matrix the loader already returns,
        because that matrix folds R0_rect into Tr and drops P2 entirely; the
        intrinsics are what the projection needs.
        """
        split, _ = (self._locate(frame.frame_id) if frame.index < 0
                    else (self.split, frame.index))
        path = self._open(split).lidar_root / "calib" / f"{frame.frame_id}.txt"
        vals = {}
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            nums = [float(v) for v in rest.split()]
            if nums:
                vals[key.strip()] = np.asarray(nums)
        P2 = vals["P2"].reshape(3, 4)
        R0 = np.eye(4); R0[:3, :3] = vals["R0_rect"].reshape(3, 3)
        Tr = np.eye(4); Tr[:3, :4] = vals["Tr_velo_to_cam"].reshape(3, 4)
        return P2, R0, Tr

    def radar_xyz_in_lidar(self, frame: Frame) -> np.ndarray:
        """Real radar detections mapped into the LiDAR frame the generator uses."""
        T = frame.calibration["radar_to_lidar"]
        h = np.concatenate([frame.radar[:, :3], np.ones((len(frame.radar), 1))], axis=1)
        return (T @ h.T).T[:, :3]
