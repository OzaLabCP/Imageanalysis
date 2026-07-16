"""Pluggable segmentation engines.

The pipeline segments a whole position by handing a list of 2-D frames to an
engine. The default ``threshold`` engine (Gaussian + Otsu + connected
components) needs no extra dependencies and runs anywhere. The optional
``cellpose`` engine runs the Cellpose deep-learning model on a GPU for far
better masks on dense / varied objects.

Each engine returns a list of int32 label images (0 = background, 1..k = cells),
already cleaned with ``relabel_min_size`` so downstream tracking/quantification
is identical regardless of engine.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np

from cellscope.analysis.segmentation import relabel_min_size, segment_frame

logger = logging.getLogger("cellscope.engines")


def resolve_device(gpu_requested: bool) -> dict:
    """Report the compute device Cellpose will actually use.

    Cellpose silently falls back to CPU when a GPU is requested but PyTorch
    cannot see one, so a run can be 10-50x slower than intended with no error.
    Callers should surface ``detail`` (and check ``fell_back``) rather than
    assume a requested GPU is active.

    Returns a dict: ``requested``, ``available``, ``device`` (cpu/cuda/mps),
    ``backend``, ``fell_back`` (GPU wanted but unavailable), and a human
    ``detail`` string.
    """
    info = {
        "requested": bool(gpu_requested),
        "available": False,
        "device": "cpu",
        "backend": None,
        "fell_back": False,
        "detail": "",
    }
    if not gpu_requested:
        info["detail"] = "GPU not requested; running on CPU."
        return info
    try:
        import torch
    except ImportError:
        info["fell_back"] = True
        info["detail"] = "PyTorch is not installed; Cellpose will run on CPU."
        return info
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            info.update(available=True, device="cuda", backend="cuda",
                        detail=f"CUDA GPU: {name}")
            return info
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            info.update(available=True, device="mps", backend="mps",
                        detail="Apple Metal (MPS) GPU")
            return info
    except Exception as exc:  # torch present but the device probe failed
        info["fell_back"] = True
        info["detail"] = f"GPU probe failed ({exc}); running on CPU."
        return info
    info["fell_back"] = True
    info["detail"] = "GPU requested but no GPU is visible to PyTorch; running on CPU."
    return info


class ThresholdEngine:
    """Gaussian smooth -> Otsu (sensitivity-nudged) -> label. CPU, no deps."""

    name = "threshold"

    def segment_stack(self, frames, settings, progress=None) -> list[np.ndarray]:
        out = []
        n = max(1, len(frames))
        for i, frame in enumerate(frames):
            out.append(segment_frame(
                frame,
                sensitivity=settings.sensitivity,
                smoothing=settings.smoothing,
                min_size=settings.min_size,
            ))
            if progress is not None:
                progress((i + 1) / n)
        return out


class CellposeEngine:
    """Cellpose deep-learning segmentation (GPU strongly recommended).

    Lazily imports ``cellpose`` and caches the loaded model per process, so a
    batch worker pays the model-load cost once. Frames are run as a batch in a
    single ``model.eval`` call so the GPU is used efficiently.
    """

    name = "cellpose"

    def __init__(self) -> None:
        self._model = None
        self._key = None
        self._legacy = False  # True if using the pre-v4 ``models.Cellpose`` API
        self.device_info: dict | None = None

    def _get_model(self, settings):
        key = (getattr(settings, "cellpose_model", "") or "", bool(settings.cellpose_gpu))
        if self._model is not None and self._key == key:
            return self._model
        from cellpose import models  # heavy import; only when actually used

        gpu = bool(settings.cellpose_gpu)
        self.device_info = resolve_device(gpu)
        if self.device_info["fell_back"]:
            # A requested-but-unavailable GPU is the single most common surprise;
            # make it loud rather than silently crawling on CPU.
            logger.warning("Cellpose: %s", self.device_info["detail"])
        else:
            logger.info("Cellpose device: %s", self.device_info["detail"])

        model_name = getattr(settings, "cellpose_model", "") or ""
        try:
            # Cellpose v4 (Cellpose-SAM): CellposeModel, cpsam by default.
            if model_name and model_name.lower() not in ("", "default", "cpsam"):
                model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
            else:
                model = models.CellposeModel(gpu=gpu)
            self._legacy = False
        except TypeError:
            # Older Cellpose API (<4): ``Cellpose`` with model_type + channels.
            model = models.Cellpose(gpu=gpu, model_type=model_name or "cyto")
            self._legacy = True
        self._model = model
        self._key = key
        return model

    def segment_stack(self, frames, settings, progress=None) -> list[np.ndarray]:
        model = self._get_model(settings)
        diameter = getattr(settings, "cellpose_diameter", None) or None
        logger.info("Cellpose: %d frame(s), diameter=%s",
                    len(frames), diameter if diameter else "model default")
        if progress is not None:
            progress(0.05)

        imgs = [np.asarray(f) for f in frames]
        # Cellpose v4 removed the ``channels`` argument; only the legacy API
        # needs it. Passing it to v4 raises TypeError, so we branch on the API
        # that ``_get_model`` actually constructed rather than guess-and-retry.
        if self._legacy:
            result = model.eval(imgs, diameter=diameter, channels=[0, 0])
        else:
            result = model.eval(imgs, diameter=diameter)
        masks = result[0] if isinstance(result, tuple) else result
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        if progress is not None:
            progress(0.9)

        out = [relabel_min_size(np.asarray(m, dtype=np.int32), settings.min_size)
               for m in masks]
        if progress is not None:
            progress(1.0)
        return out


_THRESHOLD = ThresholdEngine()
_CELLPOSE: CellposeEngine | None = None


def cellpose_available() -> bool:
    """True if the ``cellpose`` package is importable (not whether a GPU exists)."""
    try:
        return importlib.util.find_spec("cellpose") is not None
    except (ImportError, ValueError):
        return False


def available_engines() -> list[str]:
    engines = ["threshold"]
    if cellpose_available():
        engines.append("cellpose")
    return engines


def get_engine(name: str):
    """Return the engine for ``name`` ('threshold' or 'cellpose')."""
    if name == "cellpose":
        if not cellpose_available():
            raise RuntimeError(
                "Cellpose is not installed. Install it on the GPU machine with: "
                "pip install 'cellscope[cellpose]'  (and a CUDA build of torch)."
            )
        global _CELLPOSE
        if _CELLPOSE is None:
            _CELLPOSE = CellposeEngine()
        return _CELLPOSE
    return _THRESHOLD
