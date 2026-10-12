"""
Tensor Sanity Validator for FlashVSR
Detects NaN and Inf anomalies early in the pipeline to prevent wasting compute
and prevent producing black/corrupted videos.
"""

from typing import Optional
import torch


class TensorValidationError(ValueError):
    """Raised when NaN or Inf values are detected in intermediate or output tensors."""
    pass


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    allow_nan: bool = False,
    allow_inf: bool = False,
    raise_error: bool = True,
    auto_clean: bool = False,
) -> bool:
    """
    Validates that a tensor contains finite, real numbers.
    If auto_clean=True and raise_error=False, replaces NaNs and Infs with valid numbers.
    """
    if tensor is None:
        return True

    has_nan = bool(torch.isnan(tensor).any().item()) if not allow_nan else False
    has_inf = bool(torch.isinf(tensor).any().item()) if not allow_inf else False

    if has_nan or has_inf:
        err_msg = (
            f"[TE-Speed Validator] Anomaly detected in {name}: "
            f"has_nan={has_nan}, has_inf={has_inf}."
        )
        if raise_error:
            raise TensorValidationError(
                f"{err_msg} Processing was halted to prevent corrupted/black video frames."
            )
        else:
            print(f"⚠️ {err_msg}")

        if auto_clean:
            # Replace NaNs with 0.0 and Infs with clamp bounds
            tensor.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            return True

        return False

    return True
