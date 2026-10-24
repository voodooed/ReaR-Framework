"""Frame + candidates -> per-candidate material composition. Dataset-independent.

This is the module the rest of the framework calls. It closes the loop the
project's material system is built around:

    dataset -> per-point semantic label -> label space -> material mixture
            -> per-voxel composition -> permittivity -> the model input block

and it does so without knowing which dataset it is looking at. Everything
dataset-specific arrives through the adapter's config:

    materials:
      enabled: true
      label_space: cityscapes          # configs/materials/<name>.yaml
      source: camera_segmentation      # or `lidar_semantic`
      segmentation: {model: ..., short_side: 1024}
      frequency_ghz: 77.0

TWO LABEL SOURCES, ONE INTERFACE
--------------------------------
`lidar_semantic`       the dataset ships a semantic id per LiDAR point
                       (KITTI-CARLA: CARLA ground truth). No model, no camera,
                       no projection -- and `frac_labelled` is 1.0 everywhere.
`camera_segmentation`  the dataset does not, so labels are predicted from the
                       camera and transferred by projection (View-of-Delft).
                       Coverage is camera-limited: 79.6% of candidates receive
                       material, `frac_labelled` averages 0.565 over all
                       candidates.

**That asymmetry is a measured train/deploy shift, not an implementation
detail** -- see the open risk in the framework docs. It is why `frac_labelled`
is carried as an explicit feature column rather than being hidden inside the
composition: the head can see how much of a voxel was observed at all.

THE VOXEL MAP IS RECONSTRUCTED AND THEN VERIFIED
------------------------------------------------
`enumerate_candidates` does not return the point->candidate assignment, so this
module rebuilds it from the lattice definition and **checks it exactly** against
the candidate set's own dominant-semantic field before any feature is written.
A silent mis-assignment here would attach every voxel's material to the wrong
geometry while leaving every shape and summary statistic looking correct, so it
aborts rather than warns. Valid only for a fixed voxel-size schedule, which is
asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np

from rear.materials.database import MATERIAL_DB
from rear.materials.features import (MATERIAL_FEATURE_NAMES,
                                     N_MATERIAL_FEATURES,
                                     aggregate_composition,
                                     features_from_composition)
from rear.materials.taxonomy import load_label_space

SOURCE_LIDAR = "lidar_semantic"
SOURCE_CAMERA = "camera_segmentation"
SOURCES = (SOURCE_LIDAR, SOURCE_CAMERA)


def point_to_candidate(lidar_xyz: np.ndarray, centres: np.ndarray,
                       config) -> np.ndarray:
    """Index of the candidate each LiDAR point falls in, or -1.

    Reconstructs the lattice `enumerate_candidates` used, then matches each
    point's flat voxel index against the candidate centres' own. Only valid for
    `voxel_size_schedule: fixed`.
    """
    if config.voxel_size_schedule != "fixed":
        raise ValueError(
            "point_to_candidate reconstructs a FIXED voxel lattice; this "
            f"config uses voxel_size_schedule={config.voxel_size_schedule!r}. "
            "A range-adaptive schedule would need the assignment returned by "
            "enumeration itself rather than rebuilt here.")
    size = config.voxel_size
    x0, y0, z0 = config.x_range[0], config.y_range[0], config.z_range[0]
    nx = int(round((config.x_range[1] - x0) / size))
    ny = int(round((config.y_range[1] - y0) / size))
    nz = int(round((config.z_range[1] - z0) / size))

    p = np.asarray(lidar_xyz, dtype=np.float64)
    inb = ((p[:, 0] >= config.x_range[0]) & (p[:, 0] < config.x_range[1])
           & (p[:, 1] >= config.y_range[0]) & (p[:, 1] < config.y_range[1])
           & (p[:, 2] >= z0) & (p[:, 2] < config.z_range[1]))
    ix = np.clip(np.floor((p[:, 0] - x0) / size).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((p[:, 1] - y0) / size).astype(np.int64), 0, ny - 1)
    iz = np.clip(np.floor((p[:, 2] - z0) / size).astype(np.int64), 0, nz - 1)
    flat_p = (ix * ny + iy) * nz + iz

    c = np.asarray(centres, dtype=np.float64)
    cx = np.clip(np.round((c[:, 0] - x0) / size - 0.5).astype(np.int64), 0, nx - 1)
    cy = np.clip(np.round((c[:, 1] - y0) / size - 0.5).astype(np.int64), 0, ny - 1)
    cz = np.clip(np.round((c[:, 2] - z0) / size - 0.5).astype(np.int64), 0, nz - 1)
    flat_c = (cx * ny + cy) * nz + cz

    order = np.argsort(flat_c, kind="stable")
    sorted_c = flat_c[order]
    pos = np.clip(np.searchsorted(sorted_c, flat_p), 0, max(len(sorted_c) - 1, 0))
    hit = inb & (len(sorted_c) > 0) & (sorted_c[pos] == flat_p)
    out = np.full(len(p), -1, dtype=np.int64)
    out[hit] = order[pos[hit]]
    return out


def verify_mapping(voxel_index: np.ndarray, point_semantic: np.ndarray,
                   candidate_semantic: np.ndarray) -> bool:
    """The dominant class per candidate, recomputed from OUR mapping, must equal
    the candidate set's own `semantic`. Exact check, no tolerance."""
    n = len(candidate_semantic)
    keep = voxel_index >= 0
    if not keep.any():
        return True
    v = voxel_index[keep]
    s = np.asarray(point_semantic, dtype=np.int64)[keep]
    if s.min() < 0:
        raise ValueError("point semantics used for verification must be >= 0")
    cnt = np.zeros((n, int(s.max()) + 1), dtype=np.int64)
    np.add.at(cnt, (v, s), 1)
    # Ties: enumerate_candidates breaks them by lexsort on (voxel, -count),
    # keeping the SMALLEST class id among equal counts -- argmax matches.
    return bool((cnt.argmax(axis=1) == np.asarray(candidate_semantic)).all())


class MaterialExtractor:
    """Per-frame material composition for one dataset, driven by its config.

    Constructed once per run (a segmentation model, if any, is loaded lazily on
    first use and reused); `composition_for_frame` is called per frame.
    """

    def __init__(self, adapter, device: str = "cuda"):
        self.adapter = adapter
        self.config = dict(adapter.material_config())
        self.enabled = bool(self.config.get("enabled", False))
        self.label_space = load_label_space(self.config["label_space"]) \
            if self.enabled else None
        self.source = self.config.get("source", SOURCE_LIDAR)
        if self.enabled and self.source not in SOURCES:
            raise ValueError(
                f"{adapter.name}: materials.source={self.source!r} is not one of "
                f"{SOURCES}")
        self.freq_ghz = float(self.config.get("frequency_ghz",
                                              MATERIAL_DB.default_frequency_ghz))
        self.device = device
        self._segmenter = None
        self._verified = False

    # ------------------------------------------------------------------
    @property
    def n_materials(self) -> int:
        return len(MATERIAL_DB)

    @property
    def segmenter(self):
        if self._segmenter is None:
            from rear.materials.segmentation import (DEFAULT_MODEL,
                                                     DEFAULT_SHORT_SIDE,
                                                     SemanticSegmenter)
            s = dict(self.config.get("segmentation") or {})
            self._segmenter = SemanticSegmenter(
                model=s.get("model", DEFAULT_MODEL), device=self.device,
                short_side=int(s.get("short_side", DEFAULT_SHORT_SIDE)))
        return self._segmenter

    # ------------------------------------------------------------------
    def point_labels(self, frame) -> tuple[np.ndarray, np.ndarray]:
        """(N,) semantic id per LiDAR point (-1 = unobserved) and (N,) confidence.

        Confidence is 1.0 for a dataset that ships ground-truth labels and the
        segmenter's max-softmax otherwise, so the two sources are distinguishable
        downstream without a second flag.
        """
        if self.source == SOURCE_LIDAR:
            sem = frame.lidar_semantic
            if sem is None:
                raise ValueError(
                    f"{self.adapter.name} declares materials.source="
                    f"{SOURCE_LIDAR!r} but this frame carries no "
                    f"lidar_semantic. Either the adapter is not populating it "
                    f"or the config should say {SOURCE_CAMERA!r}.")
            lab = np.asarray(sem, dtype=np.int64)
            return lab, np.ones(len(lab), dtype=np.float64)

        if frame.camera is None:
            raise ValueError(
                f"{self.adapter.name} declares materials.source={SOURCE_CAMERA!r} "
                f"but frame {frame.frame_id} has no camera image.")
        P, R0, Tr = self.adapter.camera_projection(frame)
        from rear.materials.segmentation import labels_from_image
        lab_img, conf_img = self.segmenter.segment(frame.camera)
        return labels_from_image(frame.lidar[:, :3], lab_img, conf_img, P, R0, Tr)

    def composition_for_frame(self, frame, candidates, config,
                              point_semantic: np.ndarray | None = None
                              ) -> dict:
        """Per-candidate material composition for one frame.

        Returns `composition` (M, n_materials), `frac_labelled` (M,),
        `class_fractions` (M, n_classes) and `mean_confidence` (M,).
        `class_fractions` is what the `sem` control arm consumes; the production
        path uses `composition` alone, which is a sufficient statistic for the
        whole EM block.

        `point_semantic` is the per-point array that was passed to
        `enumerate_candidates` (box membership during fitting, dataset semantics
        otherwise). It is used only to VERIFY the reconstructed voxel map, once
        per extractor, and is skipped when not supplied.
        """
        n_cand = len(candidates.centres)
        n_cls = self.label_space.n_classes
        empty = dict(
            composition=np.zeros((n_cand, self.n_materials)),
            frac_labelled=np.zeros(n_cand),
            class_fractions=np.zeros((n_cand, n_cls)),
            mean_confidence=np.zeros(n_cand))
        if n_cand == 0:
            return empty

        vi = point_to_candidate(frame.lidar[:, :3], candidates.centres, config)
        if point_semantic is not None and not self._verified:
            if not verify_mapping(vi, point_semantic, candidates.semantic):
                raise RuntimeError(
                    "point->candidate mapping disagrees with enumerate_candidates' "
                    "own dominant semantic. Refusing to attach material features "
                    "to voxels that may not be the ones they were computed for.")
            self._verified = True

        labels, conf = self.point_labels(frame)
        mix = self.label_space.point_mixtures(labels)
        comp, frac = aggregate_composition(mix, vi, n_cand)

        # Class fractions and mean confidence over the LABELLED points only:
        # an unobserved point is not evidence for any class.
        keep = (vi >= 0) & (labels >= 0) & (labels < n_cls)
        cls_frac = np.zeros((n_cand, n_cls), dtype=np.float64)
        if keep.any():
            np.add.at(cls_frac, (vi[keep], labels[keep]), 1.0)
            cls_frac /= np.maximum(cls_frac.sum(axis=1, keepdims=True), 1e-12)
        n_lab = np.bincount(vi[keep], minlength=n_cand).astype(np.float64)
        c_sum = np.bincount(vi[keep], weights=conf[keep], minlength=n_cand)
        return dict(composition=comp, frac_labelled=frac,
                    class_fractions=cls_frac,
                    mean_confidence=c_sum / np.maximum(n_lab, 1e-12))

    def features_for_frame(self, frame, candidates, config,
                           point_semantic: np.ndarray | None = None) -> np.ndarray:
        """(M, 10) EM feature block for one frame -- the generation-time path."""
        d = self.composition_for_frame(frame, candidates, config, point_semantic)
        return features_from_composition(d["composition"], d["frac_labelled"],
                                         freq_ghz=self.freq_ghz)


def zero_block(n_candidates: int) -> np.ndarray:
    """The all-zero material block, for a frame with no material information.

    An all-zero row is the same thing an unlabelled candidate gets inside a
    normal frame, so a head trained with material degrades to its
    no-information behaviour rather than seeing an out-of-distribution input.
    """
    return np.zeros((n_candidates, N_MATERIAL_FEATURES), dtype=np.float64)


__all__ = ["MaterialExtractor", "MATERIAL_FEATURE_NAMES", "N_MATERIAL_FEATURES",
           "point_to_candidate", "verify_mapping", "zero_block",
           "SOURCE_LIDAR", "SOURCE_CAMERA"]
