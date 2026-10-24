"""Score a head on one frame's candidates. The single inference path.

Every consumer -- the generator, the two calibration refits, the head-level
evaluation, the qualitative ranker -- has to build exactly the same input matrix
from a frame: the physics features in the checkpoint's own variant, standardised
with the checkpoint's own constants, then the material block appended
unstandardised. Doing that in five places is how a pipeline acquires a silent
train/serve skew, so it is done here once.

    scorer = HeadScorer(adapter, architecture="prism")
    out = scorer.score(frame)          # -> p_raw, p_calibrated, rcs_dbsm, ...

Nothing in this module knows which architecture it is holding: heads are
addressed through `rear.models.base.JointHead`, and the checkpoint carries
everything needed to reproduce its own preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rear.calibration.platt import (apply_detection_calibration,
                                    load_detection_calibration)
from rear.calibration.scaling import scale_from_checkpoint
from rear.generate.generate_radar_frame import (features_for_checkpoint,
                                                material_block_for)
from rear.models.base import load_head

REPO_ROOT = Path(__file__).resolve().parents[1]


def weights_dir_for(adapter) -> Path:
    """Where this dataset's fitted artifacts live, via the `fitted_from` chain."""
    wd = Path(str(adapter.fitted("weights_dir", default=f"weights/{adapter.name}")))
    return wd if wd.is_absolute() else REPO_ROOT / wd


class HeadScorer:
    """One architecture's head, plus everything needed to feed it a frame."""

    def __init__(self, adapter, architecture: str | None = None,
                 device: str = "cpu", material_device: str | None = None,
                 with_calibration: bool = True,
                 weights_dir: Path | str | None = None):
        self.adapter = adapter
        self.device = device
        self.architecture = architecture or adapter.fitted(
            "heads", "implementation", default="prism")
        # An explicit directory exists so a smoke test cannot overwrite or read
        # a production profile by accident; default is the dataset's own.
        self.weights_dir = (Path(weights_dir) if weights_dir
                            else weights_dir_for(adapter))
        path = self.weights_dir / f"{self.architecture}_head.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"no {self.architecture!r} head at {path}. Train one with "
                f"`python scripts/train_heads.py --dataset {adapter.name} "
                f"--head {self.architecture}`.")
        self.model, self.ckpt = load_head(path, device=device)
        self.n_material = int(self.ckpt.get("n_material", 0))

        self.det_cal = None
        if with_calibration:
            p = self.weights_dir / f"detection_calibration_{self.architecture}.pkl"
            # Absent is legitimate here: the calibration REFIT scripts construct
            # a scorer precisely in order to produce this file. It is the
            # generator that refuses to run without one.
            if p.exists():
                self.det_cal = load_detection_calibration(p)

        self._extractor = None
        if self.n_material:
            from rear.materials.pipeline import MaterialExtractor
            mat = adapter.material_config()
            if not mat.get("enabled"):
                raise ValueError(
                    f"the {self.architecture!r} head expects a "
                    f"{self.n_material}-column material block but "
                    f"{adapter.name} has no `materials:` block enabled. The "
                    f"head cannot be scored on this dataset as configured.")
            self._extractor = MaterialExtractor(
                adapter, device=material_device or device)

    # ------------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return self.model.n_parameters()

    def inputs(self, frame, candidates, config,
               point_semantic: np.ndarray | None = None) -> np.ndarray:
        """(M, n_features + n_material) -- exactly what the head was trained on."""
        feats = candidates.features.astype(np.float64)
        X = scale_from_checkpoint(
            features_for_checkpoint(feats, self.ckpt, frame.ego_velocity), self.ckpt)
        if not self.n_material:
            return X
        labels, _conf = self._extractor.point_labels(frame)
        block = material_block_for(candidates, frame.lidar[:, :3], labels,
                                   self._extractor.label_space, config,
                                   self._extractor.freq_ghz, self.n_material)
        return np.hstack([X, block])

    def score(self, frame, candidates, config,
              point_semantic: np.ndarray | None = None) -> dict:
        """Detection probability (raw and calibrated) and RCS in dBsm.

        `p_calibrated` is None when no calibration has been fitted for this
        head yet -- which is the state the refit scripts run in, and is
        reported rather than silently substituted with the raw score.
        """
        X = self.inputs(frame, candidates, config, point_semantic)
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32,
                                          device=self.device))
            p = torch.sigmoid(out["det_logit"]).cpu().numpy()
            rcs = out["rcs_mean"].cpu().numpy()
        return {
            "p_raw": p,
            "p_calibrated": (apply_detection_calibration(p, self.det_cal)
                             if self.det_cal is not None else None),
            "rcs_dbsm": rcs * float(self.ckpt["y_sigma"]) + float(self.ckpt["y_mu"]),
            "rcs_standardised": rcs,
            "X": X,
        }

    def describe(self) -> dict:
        """Provenance, for a report header or a results file."""
        mat = self.ckpt.get("material", {})
        return {
            "architecture": self.architecture,
            "arm": self.ckpt.get("arm"),
            "seed": self.ckpt.get("seed"),
            "n_params": self.n_parameters,
            "feature_variant": self.ckpt.get("feature_variant"),
            "feature_names": list(self.ckpt.get("feature_names", [])),
            "n_material": self.n_material,
            "material_variant": mat.get("variant"),
            "material_label_space": mat.get("label_space"),
            "material_source": mat.get("source"),
            "weights_dir": str(self.weights_dir),
        }
