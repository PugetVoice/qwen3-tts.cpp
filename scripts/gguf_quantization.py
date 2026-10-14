"""Shared GGUF quantization, using native GGML for Q4_K encoding."""

import ctypes
from pathlib import Path

import gguf
import numpy as np


class Quantizer:
    def __init__(self, output_type: str):
        self.qtype = {
            "q4_0": gguf.GGMLQuantizationType.Q4_0,
            "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_k": gguf.GGMLQuantizationType.Q4_K,
        }.get(output_type)
        self.lib = None
        if output_type == "q4_k":
            # gguf-py implements Q4_K decoding but not encoding. Use the same
            # native implementation as the runtime instead of silently using F16.
            build_dir = Path(__file__).resolve().parents[1] / "ggml" / "build"
            candidates = [
                build_dir / "src" / "libggml-base.dylib",
                build_dir / "src" / "libggml-base.so",
                build_dir / "src" / "Release" / "ggml-base.dll",
                build_dir / "bin" / "ggml-base.dll",
            ]
            library = next((p for p in candidates if p.is_file()), None)
            if library is None:
                raise RuntimeError(
                    "Q4_K requires the native GGML shared library. Build it with:\n"
                    "cmake -S ggml -B ggml/build -DBUILD_SHARED_LIBS=ON\n"
                    "cmake --build ggml/build -j4\n"
                    "Or use --type q4_0 (no native library required)."
                )
            self.lib = ctypes.CDLL(str(library))
            self.lib.ggml_quantize_chunk.argtypes = [
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
            ]
            self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def can_quantize(self, data: np.ndarray) -> bool:
        # Only matrix weights: convolutions require floating-point kernels.
        return (self.qtype is not None and data.ndim == 2
                and data.shape[-1] % gguf.GGML_QUANT_SIZES[self.qtype][0] == 0)

    def quantize(self, data: np.ndarray) -> np.ndarray:
        if not self.can_quantize(data):
            raise ValueError(f"Unsupported quantization shape: {data.shape}")
        data = np.ascontiguousarray(data, dtype=np.float32)
        if self.lib is None:
            return gguf.quants.quantize(data, self.qtype)
        block_size, type_size = gguf.GGML_QUANT_SIZES[self.qtype]
        output = np.empty((data.shape[0], data.shape[1] // block_size * type_size), dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(
            int(self.qtype), data.ctypes.data, output.ctypes.data,
            0, data.shape[0], data.shape[1], None,
        )
        if written != output.nbytes:
            raise RuntimeError(f"GGML wrote {written} bytes; expected {output.nbytes}")
        return output
