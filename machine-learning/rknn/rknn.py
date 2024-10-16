from __future__ import annotations

from os.path import exists
import platform
from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from rknnlite.api import RKNNLite

from app.config import log

T = TypeVar("T", covariant=True)


class Newable(Protocol[T]):
    def new(self) -> None:
        ...


class _Singleton(type, Newable[T]):
    _instances: dict[_Singleton[T], Newable[T]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Newable[T]:
        if cls not in cls._instances:
            obj: Newable[T] = super(_Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = obj
        else:
            obj = cls._instances[cls]
            obj.new()
        return obj


system = platform.system()
machine = platform.machine()
os_machine = system + '-' + machine
if os_machine == 'Linux-aarch64':
    try:
        with open('/proc/device-tree/compatible') as f:
            device_compatible_str = f.read()
            rknn_support_device = ['rk3562','rk3576','rk3588','RK3566','RK3568']
            for i in rknn_support_device:
                if rknn_support_device[i] in device_compatible_str:
                    is_available = True
                    break
    except IOError:
        print('Read device node {} failed.'.format('/proc/device-tree/compatible'))
        exit(-1)



class Rknn(metaclass=_Singleton):
    def __init__(self, log_level: int = 3, tuning_level: int = 1, tuning_file: str | None = None) -> None:
        self.log_level = log_level
        self.tuning_level = tuning_level
        self.tuning_file = tuning_file
        self.output_shapes: dict[int, tuple[tuple[int], ...]] = {}
        self.input_shapes: dict[int, tuple[tuple[int], ...]] = {}
        self.rknn = RKNNLite()
        self.ann: int | None = None
        self.new()

        if self.tuning_file is not None:
            # Ensure tuning file exists (without clearing contents)
            open(self.tuning_file, "a").close()

    def new(self) -> None:
        # Set RKNN log level
        self.rknn.config(verbose=(self.log_level <= 1))  # Set verbose to True for debugging (log_level == 1)
        self.ref_count = 0

    def destroy(self) -> None:
        self.ref_count -= 1
        if self.ref_count <= 0 and self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def load(
        self,
        model_path: str,
        fast_math: bool = True,
        fp16: bool = False,
        cached_network_path: str | None = None,
    ) -> int:
        if not model_path.endswith((".rknn")):
            raise ValueError("model_path must be a file with extension .rknn")
        if not exists(model_path):
            raise ValueError("model_path must point to an existing file!")

        # Load model
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise ValueError("Cannot load model!")

        # Build model if needed
        if self.tuning_level == 0 and self.tuning_file is None:
            raise ValueError("tuning_level == 0 requires a tuning_file")
        
        if self.tuning_level > 0:
            ret = self.rknn.build(do_quantization=(self.tuning_level > 1), dataset=self.tuning_file)
            if ret != 0:
                raise RuntimeError("Failed to build RKNN model!")

        # Initialize runtime environment
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime!")

        # Get input/output shapes
        input_shapes = self.rknn.get_input_tensor_shape()
        output_shapes = self.rknn.get_output_tensor_shape()

        self.input_shapes[0] = tuple(tuple(shape) for shape in input_shapes)
        self.output_shapes[0] = tuple(tuple(shape) for shape in output_shapes)

        return 0  # return network_id (single model)

    def unload(self, network_id: int) -> None:
        self.rknn.release()

    def execute(self, network_id: int, input_tensors: list[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        if not isinstance(input_tensors, list):
            raise ValueError("input_tensors needs to be a list!")
        net_input_shapes = self.input_shapes[network_id]
        if len(input_tensors) != len(net_input_shapes):
            raise ValueError(f"input_tensors lengths {len(input_tensors)} != network inputs {len(net_input_shapes)}")
        for net_input_shape, input_tensor in zip(net_input_shapes, input_tensors):
            if net_input_shape != input_tensor.shape:
                raise ValueError(f"input_tensor shape {input_tensor.shape} != network input shape {net_input_shape}")
            if not input_tensor.flags.c_contiguous:
                raise ValueError("input_tensors must be c_contiguous numpy ndarrays")

        # Set input tensor
        ret = self.rknn.set_input_tensor(0, input_tensors[0])
        if ret != 0:
            raise RuntimeError("Failed to set input tensor")

        # Run inference
        ret = self.rknn.run()
        if ret != 0:
            raise RuntimeError("Failed to execute inference!")

        # Get output tensors
        output_tensors = [self.rknn.get_output_tensor(i) for i in range(len(self.output_shapes[network_id]))]
        return output_tensors

    def shape(self, network_id: int, input: bool = False, index: int = 0) -> tuple[int]:
        if input:
            return self.input_shapes[network_id][index]
        else:
            return self.output_shapes[network_id][index]

    def tensors(self, network_id: int, input: bool = False) -> int:
        if input:
            return len(self.input_shapes[network_id])
        else:
            return len(self.output_shapes[network_id])
