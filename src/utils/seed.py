"""
Reproducibility utilities.

Provides functions to set deterministic random seeds across
Python, NumPy, and PyTorch for experiment reproducibility.
"""

from __future__ import annotations

import os
import platform
import random
import warnings
from typing import Optional

import numpy as np
import torch


def _get_macos_major_version() -> Optional[int]:
    """Return macOS major version, e.g. 13/14, or None."""
    if platform.system() != "Darwin":
        return None
    version = platform.mac_ver()[0]
    if not version:
        return None
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None


def _is_mps_fft_supported() -> bool:
    """Project-level guard for FFT-heavy models on Apple Silicon."""
    macos_major = _get_macos_major_version()
    if macos_major is None:
        return True
    return macos_major >= 14


def _allow_unsupported_mps_override() -> bool:
    """Allow users to bypass the macOS<14 MPS safety fallback."""
    return os.getenv("FORCE_UNSUPPORTED_MPS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Models that call torch.fft in the forward path; MPS + torch.fft was flaky on macOS < 14.
MPS_FFT_GUARD_MODELS = frozenset({"TimesNet", "Autoformer"})


def _should_avoid_mps_on_old_macos_for_fft(model_name: Optional[str]) -> bool:
    """Whether to fall back to CPU instead of MPS on macOS < 14 for FFT risk.

    - If ``model_name`` is None (caller did not specify), stay conservative and avoid MPS.
    - If ``model_name`` is TimesNet or Autoformer, avoid MPS on macOS < 14.
    - All other registered models may use MPS on macOS < 14.

    Override: set env ``FORCE_UNSUPPORTED_MPS=1`` to allow MPS anyway.
    """
    if _is_mps_fft_supported():
        return False
    if _allow_unsupported_mps_override():
        return False
    if model_name is None:
        return True
    return model_name in MPS_FFT_GUARD_MODELS


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables PyTorch deterministic mode
            (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto", model_name: Optional[str] = None) -> torch.device:
    """Resolve and validate a torch device from a preference string.

    Supports: "cpu", "cuda", "cuda:<index>", "mps", and "auto".

    Behavior for "auto": prefer CUDA, then MPS, then CPU.

    On **macOS < 14**, MPS + ``torch.fft`` was unreliable. If ``model_name`` is
    ``TimesNet`` or ``Autoformer`` (or omitted), we fall back to CPU unless
    ``FORCE_UNSUPPORTED_MPS=1``. Other models may still use MPS when
    ``model_name`` is provided.

    Args:
        preference: Device preference string or a :class:`torch.device` instance.
        model_name: Registered model name (e.g. ``PatchTST``). Used only for the
            macOS < 14 MPS/FFT guard; omit for legacy conservative behavior.

    Raises:
        ValueError: when a requested device is not available or the index is invalid.

    Returns:
        torch.device instance.
    """
    # allow passing a torch.device directly
    if isinstance(preference, torch.device):
        return preference

    pref = str(preference).lower()

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            if _should_avoid_mps_on_old_macos_for_fft(model_name):
                if model_name in MPS_FFT_GUARD_MODELS:
                    warnings.warn(
                        f"Model '{model_name}' uses torch.fft; on macOS < 14 MPS may be "
                        "unreliable. Falling back to CPU. Set FORCE_UNSUPPORTED_MPS=1 to "
                        "force MPS anyway, or upgrade to macOS 14+."
                    )
                elif model_name is None:
                    warnings.warn(
                        "Detected macOS < 14 with MPS available; get_device() was called "
                        "without model_name, so falling back to CPU (conservative). "
                        "Pass model_name for selective MPS, set FORCE_UNSUPPORTED_MPS=1, "
                        "or upgrade to macOS 14+."
                    )
                return torch.device("cpu")
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            if _should_avoid_mps_on_old_macos_for_fft(model_name):
                if model_name in MPS_FFT_GUARD_MODELS:
                    warnings.warn(
                        f"MPS requested but model '{model_name}' uses torch.fft; on "
                        "macOS < 14 falling back to CPU. Set FORCE_UNSUPPORTED_MPS=1 to "
                        "force MPS anyway, or upgrade to macOS 14+."
                    )
                elif model_name is None:
                    warnings.warn(
                        "MPS requested on macOS < 14; get_device() was called without "
                        "model_name, falling back to CPU. Pass model_name for selective "
                        "MPS, set FORCE_UNSUPPORTED_MPS=1, or upgrade to macOS 14+."
                    )
                return torch.device("cpu")
            return torch.device("mps")
        raise ValueError("MPS requested but not available on this machine")

    # handle cuda and cuda:<idx>
    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but CUDA is not available")
        # allow "cuda" or "cuda:<idx>"
        if pref == "cuda":
            return torch.device("cuda:0")
        try:
            _, idx = pref.split(":", 1)
            idx = int(idx)
        except Exception as exc:  # invalid format
            raise ValueError(f"Invalid CUDA device string: {preference}") from exc
        n_gpus = torch.cuda.device_count()
        if idx < 0 or idx >= n_gpus:
            raise ValueError(f"CUDA device index {idx} out of range (0..{n_gpus-1})")
        return torch.device(f"cuda:{idx}")

    raise ValueError(f"Unrecognized device preference: {preference}")


def seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers to ensure reproducibility.

    Args:
        worker_id: Worker ID assigned by DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_state() -> dict:
    """Capture current RNG states for checkpointing.

    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch.
    """
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def set_rng_state(rng_state: dict) -> None:
    """Restore RNG states from a checkpoint (robust + defensive).

    This function accepts RNG state entries stored as:
      - `torch.ByteTensor` (CPU or CUDA)
      - list / numpy array convertible to `torch.uint8`
    and normalizes them to the expected CPU `torch.uint8` ByteTensor(s)
    before calling `torch.set_rng_state` / `torch.cuda.set_rng_state_all`.

    Args:
        rng_state: Dictionary containing saved RNG states.

    Raises:
        TypeError: if `rng_state` has an unexpected structure.
    """
    if not isinstance(rng_state, dict):
        raise TypeError(f"rng_state must be a dict, got {type(rng_state)}")

    # Python and NumPy states - unchanged
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])

    # CPU torch RNG state: ensure a CPU ByteTensor (dtype=torch.uint8)
    if "torch" in rng_state:
        state = rng_state["torch"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.uint8)
        else:
            if state.dtype != torch.uint8:
                state = state.to(dtype=torch.uint8)
            if state.device.type != "cpu":
                state = state.cpu()
        assert isinstance(state, torch.Tensor) and state.dtype == torch.uint8, (
            f"Expected CPU torch.uint8 tensor for CPU RNG state, got dtype={getattr(state, 'dtype', None)}"
        )
        torch.set_rng_state(state)

    # CUDA RNG states (list of ByteTensors) - normalize to CPU ByteTensors first
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        cuda_states = rng_state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError("torch_cuda entry must be a list/tuple of RNG states")

        processed = []
        for s in cuda_states:
            if not isinstance(s, torch.Tensor):
                t = torch.as_tensor(s, dtype=torch.uint8)
            else:
                t = s
                if t.dtype != torch.uint8:
                    t = t.to(dtype=torch.uint8)
            # set_rng_state_all expects CPU ByteTensors; convert if needed
            if t.device.type != "cpu":
                t = t.cpu()
            processed.append(t)

        # If checkpoint was created on a machine with different GPU count, trim/pad safely
        expected = torch.cuda.device_count()
        if len(processed) != expected:
            import warnings

            warnings.warn(
                f"Loaded {len(processed)} CUDA RNG states but current device_count={expected}; "
                "will use available states up to the current GPU count."
            )
            processed = processed[:expected]

        # final validation
        for p in processed:
            assert isinstance(p, torch.Tensor) and p.dtype == torch.uint8, (
                "Each CUDA RNG state must be a CPU torch.uint8 tensor"
            )

        torch.cuda.set_rng_state_all(processed)
