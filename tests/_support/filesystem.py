"""Real directory links for filesystem security tests without elevation."""

from __future__ import annotations

import ctypes
import os
import struct
from functools import cache
from pathlib import Path


@cache
def _junction_api() -> ctypes.CDLL:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True, winmode=0x00000800)
    kernel.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int32
    kernel.DeviceIoControl.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    )
    kernel.DeviceIoControl.restype = ctypes.c_int32
    return kernel


def create_directory_link(path: Path, target: Path) -> None:
    """Create a POSIX symlink or an unprivileged Windows mount-point reparse link."""
    if os.name != "nt":
        path.symlink_to(target, target_is_directory=True)
        return
    path.mkdir()
    substitute = ("\\??\\" + str(target)).encode("utf-16-le")
    printed = str(target).encode("utf-16-le")
    names = substitute + b"\0\0" + printed + b"\0\0"
    payload = (
        struct.pack(
            "<IHHHHHH",
            0xA0000003,
            8 + len(names),
            0,
            0,
            len(substitute),
            len(substitute) + 2,
            len(printed),
        )
        + names
    )
    buffer = ctypes.create_string_buffer(payload)
    kernel = _junction_api()
    handle = kernel.CreateFileW(str(path), 0x40000000, 7, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        returned = ctypes.c_uint32()
        if not kernel.DeviceIoControl(
            handle,
            0x000900A4,
            buffer,
            len(payload),
            None,
            0,
            ctypes.byref(returned),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        assert kernel.CloseHandle(handle)
