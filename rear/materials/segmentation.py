"""RGB semantic segmentation, and camera -> LiDAR label transfer.

Needed only on the FITTING side, for a dataset whose LiDAR points carry no
semantics of their own. A dataset that ships per-point labels (KITTI-CARLA does
-- CARLA ground truth) never loads a model at all; see `pipeline.py`, which
picks the source from the dataset config.

MODEL SELECTION WAS BY ELIMINATION, NOT POPULARITY
--------------------------------------------------
Default: `nvidia/segformer-b4-finetuned-cityscapes-1024-1024`, frozen, no
fine-tuning. Three constraints decided it:

1. **Dense *stuff* classes are mandatory.** 94.9% of the population that needs
   material is road, building, vegetation, terrain, pole, fence. This alone
   eliminates YOLO-seg (COCO *things* only) and SAM (class-agnostic: no
   semantics at all).
2. **Cityscapes label space.** The fitting dataset is Delft -- European urban
   street scenes, an almost exact domain match. Decisively, CARLA's palette is
   itself Cityscapes-derived, so the two datasets align natively rather than
   through an invented correspondence.
3. Mask2Former-Swin-L scores ~1 mIoU higher but adds mask-query post-processing,
   and 1 mIoU will not decide an experiment whose seed noise floor is 0.0087
   w-R2.

The model name is a CONFIG value (`materials.segmentation.model`), not a
constant, so swapping it is a YAML edit. The head's input width does not depend
on it: the EM block is 10 columns regardless of the segmenter's vocabulary.

MEASURED ON VoD: mean per-candidate confidence 0.93; ~0.4 frames/s end to end
including enumeration, so 5,562 frames is about 3.5 h on one V100. One-off,
cached.

THE KNOWN NOISE SOURCE, STATED RATHER THAN HIDDEN
-------------------------------------------------
A LiDAR point takes the label of the pixel it lands on, which is the label of
whatever is VISIBLE along that ray -- so a point on a wall behind a parked car
receives the car's label. There is no dense depth to test against and the error
is unquantified. Three mitigations are in use rather than one claim: per-point
confidence is carried, "unlabelled" stays distinguishable from "labelled as
returning nothing", and the shuffle controls bound how much of any downstream
result could be scene composition rather than material.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
DEFAULT_SHORT_SIDE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SemanticSegmenter:
    """Frozen image segmenter: RGB -> (per-pixel label, max-softmax confidence).

    Loaded lazily so importing `rear.materials` never pulls in `transformers`;
    the generation path for a dataset with native semantics must not need it.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cuda",
                 short_side: int = DEFAULT_SHORT_SIDE):
        self.model_name = model
        self.device = device
        self.short_side = int(short_side)
        self._net = None

    def _load(self):
        if self._net is None:
            import torch
            from transformers import SegformerForSemanticSegmentation
            net = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self._net = net.to(self.device).eval()
            for p in self._net.parameters():
                p.requires_grad_(False)
            self._torch = torch
        return self._net

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(H, W) int64 class ids and (H, W) float32 max-softmax confidence.

        Resize so the short side is `short_side`, ImageNet normalisation,
        per-pixel argmax, then bilinear upsample of the LOGITS back to full
        resolution before the softmax -- so confidence is the model's own, not
        an interpolation artefact of a resized label map.
        """
        torch = __import__("torch")
        net = self._load()
        H, W = image.shape[:2]
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
            t = t.permute(2, 0, 1)[None].float() / 255.0
            scale = self.short_side / min(H, W)
            t = torch.nn.functional.interpolate(
                t, size=(int(round(H * scale)), int(round(W * scale))),
                mode="bilinear", align_corners=False)
            t = (t - mean) / std
            logits = net(pixel_values=t).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False)
            conf, lab = logits.softmax(dim=1).max(dim=1)
        return (lab[0].cpu().numpy().astype(np.int64),
                conf[0].cpu().numpy().astype(np.float32))


def project_to_image(xyz_lidar: np.ndarray, P: np.ndarray, R0: np.ndarray,
                     Tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LiDAR-frame points -> (u, v, depth) in the camera image.

        image = P @ R0_rect @ Tr_velo_to_cam @ X_lidar

    straight from the dataset's own calibration files. `depth <= 0` means the
    point is behind the camera and carries no label.
    """
    xyz = np.asarray(xyz_lidar, dtype=np.float64)
    h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    cam = (np.asarray(R0) @ np.asarray(Tr) @ h.T).T
    depth = cam[:, 2]
    uvw = (np.asarray(P) @ cam.T).T
    w = uvw[:, 2]
    safe = np.maximum(np.abs(w), 1e-6) * np.sign(np.where(w == 0, 1.0, w))
    return uvw[:, 0] / safe, uvw[:, 1] / safe, depth


def labels_from_image(xyz_lidar: np.ndarray, label_image: np.ndarray,
                      conf_image: np.ndarray, P: np.ndarray, R0: np.ndarray,
                      Tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (label, confidence) by projection. Label -1 = no information.

    -1 is not a class and is never mapped to a material: a point outside the
    camera frustum is unobserved, which is a different statement from a point
    the segmenter labelled `sky`.
    """
    u, v, depth = project_to_image(xyz_lidar, P, R0, Tr)
    H, W = label_image.shape[:2]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    ok = ((depth > 0) & np.isfinite(u) & np.isfinite(v)
          & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H))
    lab = np.full(len(xyz_lidar), -1, dtype=np.int64)
    conf = np.zeros(len(xyz_lidar), dtype=np.float64)
    lab[ok] = label_image[vi[ok], ui[ok]]
    conf[ok] = conf_image[vi[ok], ui[ok]]
    return lab, conf
