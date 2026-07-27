# Copyright (c) Opendatalab. All rights reserved.
import os
import sys

from loguru import logger

from mineru.backend.vlm.utils import set_lmdeploy_backend
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def main():
    args = sys.argv[1:]

    has_port_arg = False
    has_gpu_memory_utilization_arg = False
    has_log_level_arg = False
    device_type = ""
    lm_backend = ""

    # Validate the current value.
    indices_to_remove = []

    for i, arg in enumerate(args):
        if arg == "--server-port" or arg.startswith("--server-port="):
            has_port_arg = True
        if arg == "--cache-max-entry-count" or arg.startswith("--cache-max-entry-count="):
            has_gpu_memory_utilization_arg = True
        if arg == "--log-level" or arg.startswith("--log-level="):
            has_log_level_arg = True
        if arg == "--backend" or arg == "--lmdeploy-backend":
            if i + 1 < len(args):
                lm_backend = args[i + 1]
                indices_to_remove.extend([i, i + 1])
        elif arg.startswith("--backend=") or arg.startswith("--lmdeploy-backend="):
            lm_backend = arg.split("=", 1)[1]
            indices_to_remove.append(i)
        if arg == "--device" or arg == "--lmdeploy-device":
            if i + 1 < len(args):
                device_type = args[i + 1]
                indices_to_remove.extend([i, i + 1])
        elif arg.startswith("--device=") or arg.startswith("--lmdeploy-device="):
            device_type = arg.split("=", 1)[1]
            indices_to_remove.append(i)

    # Remove invalid or unnecessary data.
    for i in sorted(set(indices_to_remove), reverse=True):
        args.pop(i)

    # Add the value to the result.
    if not has_port_arg:
        args.extend(["--server-port", "30000"])
    if not has_gpu_memory_utilization_arg:
        args.extend(["--cache-max-entry-count", "0.5"])
    if not has_log_level_arg:
        args.extend(["--log-level", "ERROR"])

    device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", device_type)
    if device_type == "":
        device_type = "cuda"
    elif device_type not in ["cuda", "ascend", "maca", "camb"]:
        raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
    lm_backend = os.getenv("MINERU_LMDEPLOY_BACKEND", lm_backend)
    if lm_backend == "":
        lm_backend = set_lmdeploy_backend(device_type)
    elif lm_backend not in ["pytorch", "turbomind"]:
        raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")
    logger.info(f"lmdeploy device is: {device_type}, lmdeploy backend is: {lm_backend}")

    if lm_backend == "pytorch":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    args.extend(["--device", device_type])
    args.extend(["--backend", lm_backend])

    model_path = auto_download_and_get_model_root_path("/", "vlm")

    # logger.debug(args)

    # Configure the model.
    sys.argv = [sys.argv[0]] + ["serve", "api_server", model_path] + args

    if os.getenv('OMP_NUM_THREADS') is None:
        os.environ["OMP_NUM_THREADS"] = "1"

    # Process the service request.
    print(f"start lmdeploy server: {sys.argv}")

    # Process the service request.
    os.system("lmdeploy " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
