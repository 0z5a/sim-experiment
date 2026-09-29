"""Exercise CUDA's two resolver ABIs and a real device round-trip."""

import ctypes as C
import json


def main() -> None:
    cuda = C.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [C.c_uint]
    assert cuda.cuInit(0) == 0
    v1 = cuda.cuGetProcAddress
    v1.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.c_int, C.c_uint64]
    v2 = cuda.cuGetProcAddress_v2
    v2.argtypes = v1.argtypes + [C.POINTER(C.c_int)]
    pointer = C.c_void_p()
    status = C.c_int(-1)
    assert v1(b"cuDeviceGetCount", C.byref(pointer), 11030, 0) == 0
    count = C.c_int()
    assert C.CFUNCTYPE(C.c_int, C.POINTER(C.c_int))(pointer.value)(C.byref(count)) == 0
    assert count.value == 1, count.value
    assert v2(b"cuDeviceGetCount", C.byref(pointer), 12000, 0, C.byref(status)) == 0
    assert status.value == 0 and pointer.value
    missing = v2(b"cuThisSymbolDoesNotExist", C.byref(pointer), 12000, 0, C.byref(status))
    assert not pointer.value and status.value != 0
    import torch
    value = torch.arange(8, device="cuda", dtype=torch.int64)
    assert (value + 1).cpu().tolist() == list(range(1, 9))
    print(json.dumps({"devices": count.value, "missing_symbol_result": missing, "missing_symbol_status": status.value, "round_trip": "passed"}))


if __name__ == "__main__":
    main()
