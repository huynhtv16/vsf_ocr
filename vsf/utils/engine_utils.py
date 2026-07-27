# Copyright (c) Opendatalab. All rights reserved.
import os

from loguru import logger

from vsf.utils.check_sys_env import is_mac_os_version_supported, is_windows_environment, is_mac_environment, \
    is_linux_environment


def get_vlm_engine(inference_engine: str, is_async: bool = False) -> str:
    """
    Validate the current value.

    Args:
        Implementation detail.
        Implementation detail.

    Returns:
        Implementation detail.
    """
    if inference_engine == 'auto':
        # Implementation detail.
        if is_windows_environment():
            inference_engine = _select_windows_engine()
        elif is_linux_environment():
            inference_engine = _select_linux_engine(is_async)
        elif is_mac_environment():
            inference_engine = _select_mac_engine()
        else:
            logger.warning("Unknown operating system, falling back to transformers")
            inference_engine = 'transformers'

    formatted_engine = _format_engine_name(inference_engine)
    logger.info(f"Using {formatted_engine} as the inference engine for VLM.")
    return formatted_engine


def _select_windows_engine() -> str:
    """Implementation detail."""
    try:
        import lmdeploy
        return 'lmdeploy'
    except ImportError:
        return 'transformers'


def _select_linux_engine(is_async: bool) -> str:
    """Implementation detail."""
    try:
        import vllm
        return 'vllm-async' if is_async else 'vllm'
    except ImportError:
        try:
            import lmdeploy
            return 'lmdeploy'
        except ImportError:
            return 'transformers'


def _select_mac_engine() -> str:
    """Implementation detail."""
    try:
        from mlx_vlm import load as mlx_load
        if is_mac_os_version_supported():
            return 'mlx'
        else:
            return 'transformers'
    except ImportError:
        return 'transformers'


def _format_engine_name(engine: str) -> str:
    """Implementation detail."""
    if engine != 'transformers':
        return f"{engine}-engine"
    return engine
