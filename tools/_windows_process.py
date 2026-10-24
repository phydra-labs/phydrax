# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Race-free, host-only Win32 child ownership for accelerator command evidence.

CreateProcessW starts suspended; an unnamed, non-inherited kill-on-close Job
Object owns the child before its first instruction. Only duplicated standard
handles are inherited. No breakaway flags or shell are used. Closing the job on
supervisor exit is a final safety net, not a substitute for observed teardown.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, ExitStack
from functools import cache
from pathlib import Path


_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_LIST = 0x00020002
_KILL_ON_JOB_CLOSE = 0x00002000
_WAIT_TIMEOUT = 258
_WAIT_FAILED = 0xFFFFFFFF
_TERMINATED = 124
_CLEANUP_SECONDS = 10.0


class _StartupInfo(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.c_uint32),
        ("dwY", ctypes.c_uint32),
        ("dwXSize", ctypes.c_uint32),
        ("dwYSize", ctypes.c_uint32),
        ("dwXCountChars", ctypes.c_uint32),
        ("dwYCountChars", ctypes.c_uint32),
        ("dwFillAttribute", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("wShowWindow", ctypes.c_uint16),
        ("cbReserved2", ctypes.c_uint16),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", ctypes.c_void_p),
        ("hStdOutput", ctypes.c_void_p),
        ("hStdError", ctypes.c_void_p),
    )


class _StartupInfoEx(ctypes.Structure):
    _fields_ = (("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p))


class _ProcessInformation(ctypes.Structure):
    _fields_ = (
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_uint32),
        ("dwThreadId", ctypes.c_uint32),
    )


class _BasicLimit(ctypes.Structure):
    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    )


class _ExtendedLimit(ctypes.Structure):
    _fields_ = (
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _BasicAccounting(ctypes.Structure):
    _fields_ = (
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    )


@cache
def _kernel32() -> ctypes.CDLL:
    if os.name != "nt":
        raise OSError("Windows process supervision requires Win32")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True, winmode=0x00000800)
    handle = ctypes.c_void_p
    dword = ctypes.c_uint32
    boolean = ctypes.c_int32
    kernel.CreateJobObjectW.argtypes = (handle, ctypes.c_wchar_p)
    kernel.CreateJobObjectW.restype = handle
    kernel.SetInformationJobObject.argtypes = (handle, ctypes.c_int32, handle, dword)
    kernel.SetInformationJobObject.restype = boolean
    kernel.QueryInformationJobObject.argtypes = (
        handle,
        ctypes.c_int32,
        handle,
        dword,
        ctypes.POINTER(dword),
    )
    kernel.QueryInformationJobObject.restype = boolean
    kernel.AssignProcessToJobObject.argtypes = (handle, handle)
    kernel.AssignProcessToJobObject.restype = boolean
    kernel.TerminateJobObject.argtypes = (handle, dword)
    kernel.TerminateJobObject.restype = boolean
    kernel.TerminateProcess.argtypes = (handle, dword)
    kernel.TerminateProcess.restype = boolean
    kernel.CloseHandle.argtypes = (handle,)
    kernel.CloseHandle.restype = boolean
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = handle
    kernel.DuplicateHandle.argtypes = (
        handle,
        handle,
        handle,
        ctypes.POINTER(handle),
        dword,
        boolean,
        dword,
    )
    kernel.DuplicateHandle.restype = boolean
    kernel.InitializeProcThreadAttributeList.argtypes = (
        handle,
        dword,
        dword,
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel.InitializeProcThreadAttributeList.restype = boolean
    kernel.UpdateProcThreadAttribute.argtypes = (
        handle,
        dword,
        ctypes.c_size_t,
        handle,
        ctypes.c_size_t,
        handle,
        handle,
    )
    kernel.UpdateProcThreadAttribute.restype = boolean
    kernel.DeleteProcThreadAttributeList.argtypes = (handle,)
    kernel.DeleteProcThreadAttributeList.restype = None
    kernel.CreateProcessW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        handle,
        handle,
        boolean,
        dword,
        handle,
        ctypes.c_wchar_p,
        ctypes.POINTER(_StartupInfoEx),
        ctypes.POINTER(_ProcessInformation),
    )
    kernel.CreateProcessW.restype = boolean
    kernel.ResumeThread.argtypes = (handle,)
    kernel.ResumeThread.restype = dword
    kernel.WaitForSingleObject.argtypes = (handle, dword)
    kernel.WaitForSingleObject.restype = dword
    kernel.GetExitCodeProcess.argtypes = (handle, ctypes.POINTER(dword))
    kernel.GetExitCodeProcess.restype = boolean
    return kernel


def _error(operation: str) -> OSError:
    error = ctypes.WinError(ctypes.get_last_error())
    error.add_note(operation)
    return error


def _close(kernel: ctypes.CDLL, handle: int) -> None:
    if not kernel.CloseHandle(handle):
        raise _error("CloseHandle")


def _duplicate(kernel: ctypes.CDLL, handle: int, *, inherit: bool) -> int:
    duplicate = ctypes.c_void_p()
    current = kernel.GetCurrentProcess()
    if not kernel.DuplicateHandle(
        current, handle, current, ctypes.byref(duplicate), 0, inherit, 2
    ):
        raise _error("DuplicateHandle")
    if duplicate.value is None:
        raise OSError("DuplicateHandle returned no handle")
    return duplicate.value


@contextmanager
def _startup(
    kernel: ctypes.CDLL, stdout_fd: int, stderr_fd: int
) -> Iterator[_StartupInfoEx]:
    import msvcrt

    with ExitStack() as stack:
        stdin = stack.enter_context(open(os.devnull, "rb"))
        handles: list[int] = []
        for fd in (stdin.fileno(), stdout_fd, stderr_fd):
            handle = _duplicate(kernel, msvcrt.get_osfhandle(fd), inherit=True)
            stack.callback(_close, kernel, handle)
            handles.append(handle)
        size = ctypes.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        if not size.value:
            raise _error("InitializeProcThreadAttributeList size query")
        attributes = ctypes.create_string_buffer(size.value)
        if not kernel.InitializeProcThreadAttributeList(
            attributes, 1, 0, ctypes.byref(size)
        ):
            raise _error("InitializeProcThreadAttributeList")
        stack.callback(kernel.DeleteProcThreadAttributeList, attributes)
        inherited = (ctypes.c_void_p * len(handles))(*handles)
        if not kernel.UpdateProcThreadAttribute(
            attributes, 0, _HANDLE_LIST, inherited, ctypes.sizeof(inherited), None, None
        ):
            raise _error("UpdateProcThreadAttribute handle allowlist")
        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        (
            startup.StartupInfo.hStdInput,
            startup.StartupInfo.hStdOutput,
            startup.StartupInfo.hStdError,
        ) = handles
        startup.lpAttributeList = ctypes.addressof(attributes)
        yield startup


def _environment_block(environment: Mapping[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    entries: list[str] = []
    for name, value in sorted(environment.items(), key=lambda item: item[0].upper()):
        # Windows also permits inherited drive-current-directory keys such as =C:.
        if not name or "=" in name[1:] or "\0" in name or "\0" in value:
            raise ValueError("Invalid Windows environment entry")
        entries.append(f"{name}={value}")
    return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")


def _assign_job(kernel: ctypes.CDLL, job: int, process: int) -> None:
    if not kernel.AssignProcessToJobObject(job, process):
        raise _error("AssignProcessToJobObject; child remains suspended")


def _wait(kernel: ctypes.CDLL, process: int, deadline: float) -> bool:
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        milliseconds = min(math.ceil(remaining * 1000.0), _WAIT_FAILED - 1)
        result = kernel.WaitForSingleObject(process, milliseconds)
        if result == 0:
            return True
        if result == _WAIT_FAILED:
            raise _error("WaitForSingleObject")
        if result != _WAIT_TIMEOUT:
            raise OSError(f"Unexpected process wait status {result}")
        if time.monotonic() >= deadline:
            return False


def _terminate_suspended(kernel: ctypes.CDLL, process: int) -> None:
    if not kernel.TerminateProcess(process, _TERMINATED):
        raise _error("TerminateProcess for unassigned suspended child")
    if not _wait(kernel, process, time.monotonic() + _CLEANUP_SECONDS):
        raise TimeoutError("Unassigned suspended child did not terminate")


def _terminate_job(kernel: ctypes.CDLL, job: int) -> None:
    if not kernel.TerminateJobObject(job, _TERMINATED):
        raise _error("TerminateJobObject")
    deadline = time.monotonic() + _CLEANUP_SECONDS
    while True:
        accounting = _BasicAccounting()
        if not kernel.QueryInformationJobObject(
            job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise _error("QueryInformationJobObject during teardown")
        if accounting.ActiveProcesses == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned Windows job descendants did not terminate")
        time.sleep(0.01)


def run_in_job(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    stdout_fd: int,
    stderr_fd: int,
) -> tuple[int, bool]:
    """Return the exact unsigned DWORD exit status and deadline observation.

    Borrow the caller's log descriptors; duplicate only handles the child needs.
    Own every new handle until closure. Any launch, assignment, wait, status, or
    teardown failure raises, so unknown cleanup cannot be recorded as success.
    A supervisor crash closes its non-inherited job handle in the OS.
    """
    if not command:
        raise ValueError("A command must contain an executable")
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError("timeout must be finite and non-negative")
    kernel = _kernel32()
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
    block = _environment_block(environment)
    deadline = time.monotonic() + timeout
    with ExitStack() as stack:
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise _error("CreateJobObjectW")
        stack.callback(_close, kernel, job)
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise _error("SetInformationJobObject kill-on-close")
        with _startup(kernel, stdout_fd, stderr_fd) as startup:
            info = _ProcessInformation()
            if not kernel.CreateProcessW(
                None,
                command_line,
                None,
                None,
                True,
                _CREATE_SUSPENDED
                | _CREATE_UNICODE_ENVIRONMENT
                | _EXTENDED_STARTUPINFO_PRESENT,
                block,
                str(cwd),
                ctypes.byref(startup),
                ctypes.byref(info),
            ):
                raise _error("CreateProcessW suspended")
            stack.callback(_close, kernel, info.hProcess)
            stack.callback(_close, kernel, info.hThread)
            try:
                _assign_job(kernel, job, info.hProcess)
            except BaseException:
                _terminate_suspended(kernel, info.hProcess)
                raise
            # Registered before closing startup resources or resuming: any later
            # failure still terminates and observes this entire owned job.
            stack.callback(_terminate_job, kernel, job)
        if kernel.ResumeThread(info.hThread) == _WAIT_FAILED:
            raise _error("ResumeThread")
        timed_out = not _wait(kernel, info.hProcess, deadline)
        if timed_out:
            _terminate_job(kernel, job)
            if not _wait(kernel, info.hProcess, time.monotonic() + _CLEANUP_SECONDS):
                raise TimeoutError("Owned process did not signal after job teardown")
        status = ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(status)):
            raise _error("GetExitCodeProcess")
        # A signaled process may legitimately exit with STILL_ACTIVE (259).
        # Its wait result, rather than a special-cased code, establishes exit.
        return status.value, timed_out
