# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path

import torch

from .modeling.architectures.base_model import BaseModel


# Implementation detail.
OCR_INFERENCE_PRECISION = "auto"


class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.ocr_inference_dtype = torch.float32
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def _resolve_inference_dtype(self, device):
        """Parse the input data."""
        precision = OCR_INFERENCE_PRECISION.lower()
        device_name = str(device).lower()
        is_cpu = device_name.startswith("cpu")

        if precision not in {"auto", "fp32", "fp16"}:
            raise ValueError(
                "OCR_INFERENCE_PRECISION must be one of: auto, fp32, fp16"
            )
        if precision == "fp32" or is_cpu:
            return torch.float32
        return torch.float16

    def _apply_inference_precision(self, device):
        """Implementation detail."""
        self.net.to(device)
        self.ocr_inference_dtype = self._resolve_inference_dtype(device)
        if self.ocr_inference_dtype == torch.float16:
            self.net.to(dtype=torch.float16)

    def _to_inference_dtype(self, tensor):
        """Convert the value to the required format."""
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
            inference_dtype = getattr(self, "ocr_inference_dtype", torch.float32)
            return tensor.to(dtype=inference_dtype)
        return tensor

    @staticmethod
    def _is_safetensors_path(weights_path):
        """Validate the current value."""
        return Path(weights_path).suffix == ".safetensors"

    @staticmethod
    def _load_weight_file(weights_path):
        """Process the file path."""
        if BaseOCRV20._is_safetensors_path(weights_path):
            from safetensors.torch import load_file

            return load_file(str(weights_path), device="cpu")
        try:
            return torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(weights_path, map_location="cpu")

    @staticmethod
    def _normalize_ppocrv6_state_dict(weights, weights_path):
        """Implementation detail."""
        if not BaseOCRV20._is_safetensors_path(weights_path):
            return weights
        if not any(key.startswith("model.") for key in weights.keys()):
            return weights
        return {
            key.removeprefix("model."): value
            for key, value in weights.items()
        }

    def read_pytorch_weights(self, weights_path):
        """Extract the required value."""
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = self._load_weight_file(weights_path)
        return self._normalize_ppocrv6_state_dict(weights, weights_path)

    def get_out_channels(self, weights):
        """Prepare the output value."""
        if "head.head.weight" in weights:
            # Implementation detail.
            return weights["head.head.weight"].shape[0]
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        # print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        """Implementation detail."""
        self.net.load_state_dict(self.read_pytorch_weights(weights_path))
        # print('model is loaded: {}'.format(weights_path))

    def inference(self, inputs):
        with torch.inference_mode():
            infer = self.net(inputs)
        return infer
