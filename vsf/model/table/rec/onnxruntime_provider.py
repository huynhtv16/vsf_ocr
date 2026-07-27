# Copyright (c) Opendatalab. All rights reserved.
from typing import Any, List, Sequence, Tuple

from loguru import logger

from mineru.utils.config_reader import get_device


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER_OPTS = {
    "arena_extend_strategy": "kSameAsRequested",
}
CUDA_PROVIDER_OPTS = {
    "cudnn_conv_algo_search": "HEURISTIC",
}


def _normalize_device(device: object) -> str:
    """Implementation detail."""
    if not isinstance(device, str):
        return ""
    return device.split(":", 1)[0].strip().lower()


def _build_cpu_provider() -> Tuple[str, dict[str, Any]]:
    """Build the required output."""
    return (CPU_PROVIDER, dict(CPU_PROVIDER_OPTS))


def _build_cuda_provider() -> Tuple[str, dict[str, Any]]:
    return (CUDA_PROVIDER, dict(CUDA_PROVIDER_OPTS))


def build_table_onnx_providers(
    available_providers: Sequence[str],
) -> List[Tuple[str, dict[str, Any]]]:
    """Process table content."""
    cpu_provider = _build_cpu_provider()
    cuda_provider = _build_cuda_provider()
    device = _normalize_device(get_device())

    # Implementation detail.
    if device != "cuda":
        return [cpu_provider]

    if CUDA_PROVIDER in available_providers:
        return [cuda_provider, cpu_provider]

    return [cpu_provider]
