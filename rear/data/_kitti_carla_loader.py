"""Load KITTI-CARLA frames (LiDAR, RGB, ego velocity, calibration).

Verified against an actual local download of Town01/Town02 (see Step 1
exploration report). Directory layout, per one town:

    <town_root>/generated/frames/frame_NNNN.ply           # binary PLY, one per frame
    <town_root>/generated/images_rgb/NNNN_{0,1}.png        # stereo L/R, RGBA
    <town_root>/generated/full_poses_lidar.txt             # 1 kHz ego pose, [R|t|timestamp] per row
    <town_root>/generated/full_ts_camera.txt               # 10 Hz frame_id -> timestamp map
    <town_root>/generated/lidar_to_cam0.txt                # static [R|t] extrinsic, 3x4

Frame indices correspond 1:1 across lidar/rgb (confirmed: both run
0000..NNNN contiguously, same count) -- frame i's LiDAR is
`frame_{i:04d}.ply` and its left camera image is `{i:04d}_0.png`.

Two things the raw files don't match a naive KITTI-style assumption:
  - `cos_angle_lidar_surface` (the per-point incidence-angle cosine) is
    in [0, 1], not [-1, 1] -- points that weren't hit face-on simply
    aren't in the file, so a negative cosine never occurs.
  - RGB/depth PNGs are RGBA, not RGB; the alpha channel is dropped here.

`lidar_to_cam0.txt`'s rotation block is the identity matrix -- it
expresses cam0 in the *vehicle/LiDAR* axis convention (x-forward,
y-left, z-up), not the optical convention (x-right, y-down, z-forward)
a pinhole projection needs. This loader returns the extrinsic exactly
as stored; any optical-frame remap belongs at the point of use (see
scripts/verify_kitti_carla.py), since the loader's job is to report
what's on disk, not to bake in a projection-specific assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData


@dataclass
class KittiCarlaFrame:
    lidar: np.ndarray            # (N, 4) float32: x, y, z, cos_angle
    lidar_semantic: np.ndarray   # (N,) int64: semantic class per point
    rgb: np.ndarray              # (H, W, 3) uint8: left camera image
    ego_velocity: np.ndarray     # (3,) float32, m/s
    calibration: dict            # {'lidar_to_cam0': (4, 4) float64 np.ndarray}
    # Per-point CARLA instance id. Combined with lidar_semantic this yields 3-D
    # boxes (see data/kc_boxes.py) -- KITTI-CARLA ships no box annotations.
    lidar_instance: np.ndarray | None = None


def _load_lidar_to_cam(path: Path) -> np.ndarray:
    """Parse a `lidar_to_camN.txt` file (one data line: 9 rotation + 3
    translation entries, row-major) into a 4x4 homogeneous matrix."""
    with open(path) as f:
        lines = [l for l in f if not l.strip().startswith("#") and l.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one data line in {path}, found {len(lines)}")
    values = [float(v) for v in lines[0].split()]
    if len(values) != 12:
        raise ValueError(f"Expected 12 values (3x4 [R|t]) in {path}, found {len(values)}")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = np.array(values, dtype=np.float64).reshape(3, 4)
    return mat


def _load_pose_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse `full_poses_lidar.txt` (1 header line, then rows of
    9 rotation + 3 translation + 1 timestamp). Returns (timestamps (M,),
    positions (M, 3)) -- rotation is discarded, only position is needed
    for velocity."""
    data = np.loadtxt(path, comments="#")
    if data.shape[1] != 13:
        raise ValueError(f"Expected 13 columns in {path}, found {data.shape[1]}")
    positions = data[:, [3, 7, 11]]
    timestamps = data[:, 12]
    return timestamps, positions


def _load_frame_timestamps(path: Path) -> np.ndarray:
    """Parse `full_ts_camera.txt` (1 header line, then `frame_id
    timestamp` rows). Returns timestamps indexed by frame id."""
    data = np.loadtxt(path, comments="#")
    frame_ids = data[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(len(frame_ids))):
        raise ValueError(f"{path}: frame ids are not a contiguous 0..N-1 range")
    return data[:, 1]


class KittiCarlaDataset:
    def __init__(self, town_root: str, load_rgb: bool = True):
        self.town_root = Path(town_root)
        self.load_rgb = load_rgb
        gen = self.town_root / "generated"

        self.frames_dir = gen / "frames"
        self.rgb_dir = gen / "images_rgb"
        required = {
            "frames": self.frames_dir,
            "images_rgb": self.rgb_dir,
            "full_poses_lidar.txt": gen / "full_poses_lidar.txt",
            "full_ts_camera.txt": gen / "full_ts_camera.txt",
            "lidar_to_cam0.txt": gen / "lidar_to_cam0.txt",
        }
        missing = [name for name, p in required.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing expected KITTI-CARLA paths under {gen}: {missing}")

        self._lidar_to_cam0 = _load_lidar_to_cam(required["lidar_to_cam0.txt"])

        self._num_frames = len(sorted(self.frames_dir.glob("frame_*.ply")))
        if self._num_frames == 0:
            raise FileNotFoundError(f"No frame_*.ply files found in {self.frames_dir}")

        pose_ts, pose_pos = _load_pose_table(required["full_poses_lidar.txt"])
        frame_ts = _load_frame_timestamps(required["full_ts_camera.txt"])
        if len(frame_ts) != self._num_frames:
            raise ValueError(
                f"full_ts_camera.txt has {len(frame_ts)} rows but {self._num_frames} "
                f"lidar frames were found -- frame/timestamp count mismatch, refusing to guess."
            )
        self._ego_velocity = self._compute_velocities(frame_ts, pose_ts, pose_pos)

    @staticmethod
    def _nearest_pose_positions(query_ts: np.ndarray, pose_ts: np.ndarray, pose_pos: np.ndarray) -> np.ndarray:
        """Vectorized nearest-timestamp lookup. `pose_ts` must be sorted
        ascending (true of full_poses_lidar.txt: monotonically increasing
        at 1 kHz)."""
        idx = np.searchsorted(pose_ts, query_ts)
        idx = np.clip(idx, 1, len(pose_ts) - 1)
        left, right = idx - 1, idx
        use_left = np.abs(pose_ts[left] - query_ts) <= np.abs(pose_ts[right] - query_ts)
        chosen = np.where(use_left, left, right)
        return pose_pos[chosen]

    def _compute_velocities(self, frame_ts: np.ndarray, pose_ts: np.ndarray, pose_pos: np.ndarray) -> np.ndarray:
        """velocity[i] = (pos(t[i+1]) - pos(t[i])) / (t[i+1] - t[i]), i.e.
        forward difference between frame i and i+1's nearest 1 kHz pose,
        as specified. Two boundary cases the spec doesn't fully cover
        symmetrically:
          - frame 0: no forward difference is undefined here (i=0 has a
            valid i+1), but the spec explicitly asks for frame 0 to copy
            frame 1's velocity anyway, so we override it after computing.
          - last frame: has no i+1, so it has no forward difference by
            construction; it reuses the last defined one (frames
            N-2 -> N-1), the same "copy the neighbor" rule the spec
            applies at the start, applied at the other end.
        """
        n = len(frame_ts)
        positions = self._nearest_pose_positions(frame_ts, pose_ts, pose_pos)

        velocity = np.zeros((n, 3), dtype=np.float32)
        dt = frame_ts[1:] - frame_ts[:-1]
        dpos = positions[1:] - positions[:-1]
        velocity[:-1] = (dpos / dt[:, None]).astype(np.float32)
        velocity[-1] = velocity[-2]   # last frame: no i+1, reuse previous
        velocity[0] = velocity[1]     # explicit spec edge case
        return velocity

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> KittiCarlaFrame:
        if not (0 <= idx < self._num_frames):
            raise IndexError(f"Frame index {idx} out of range [0, {self._num_frames})")

        ply = PlyData.read(self.frames_dir / f"frame_{idx:04d}.ply")
        vertex = ply["vertex"]
        lidar = np.stack(
            [vertex["x"], vertex["y"], vertex["z"], vertex["cos_angle_lidar_surface"]], axis=-1
        ).astype(np.float32)
        lidar_semantic = np.asarray(vertex["semantic"]).astype(np.int64)
        lidar_instance = np.asarray(vertex["instance"]).astype(np.int64)

        # Decoding the camera PNG dominates load time and is irrelevant to the
        # radar pipeline, so it is skippable.
        if self.load_rgb:
            rgb = np.array(Image.open(self.rgb_dir / f"{idx:04d}_0.png").convert("RGB"))
        else:
            rgb = None

        return KittiCarlaFrame(
            lidar=lidar,
            lidar_semantic=lidar_semantic,
            lidar_instance=lidar_instance,
            rgb=rgb,
            ego_velocity=self._ego_velocity[idx],
            calibration={"lidar_to_cam0": self._lidar_to_cam0},
        )
