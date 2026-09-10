from __future__ import annotations

import contextlib
import errno
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from filelock import FileLock, Timeout, lock_descriptor, unlock_descriptor

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:  # pragma: win32 cover
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    _ERROR_LOCK_VIOLATION = 33
    _win_ctypes = cast(Any, ctypes)
    _win_msvcrt = cast(Any, msvcrt)

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    _kernel32 = _win_ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL

    def _lock_readonly_descriptor(descriptor: int) -> bool:
        overlapped = _Overlapped()
        handle = _win_msvcrt.get_osfhandle(descriptor)
        if _kernel32.LockFileEx(
            handle,
            _LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            ctypes.byref(overlapped),
        ):
            return True
        error = _win_ctypes.get_last_error()
        if error == _ERROR_LOCK_VIOLATION:
            return False
        raise _win_ctypes.WinError(error)

    def _unlock_readonly_descriptor(descriptor: int) -> None:
        overlapped = _Overlapped()
        handle = _win_msvcrt.get_osfhandle(descriptor)
        if not _kernel32.UnlockFileEx(
            handle,
            0,
            1,
            0,
            ctypes.byref(overlapped),
        ):  # pragma: no cover
            raise _win_ctypes.WinError(_win_ctypes.get_last_error())

else:

    def _lock_readonly_descriptor(descriptor: int) -> bool:
        return lock_descriptor(descriptor, blocking=False)

    def _unlock_readonly_descriptor(descriptor: int) -> None:
        unlock_descriptor(descriptor)


class AuthorizationGuardUnavailable(RuntimeError):
    """Raised when the authorization guard cannot be acquired safely."""


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _is_redirect(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or _is_reparse_point(file_stat)


class AuthorizationGuard:
    """Coordinate provider access with writable knowledge control operations."""

    def __init__(self, path: Path):
        expanded = path.expanduser()
        absolute = Path(os.path.abspath(os.fspath(expanded)))
        try:
            parent = absolute.parent.resolve(strict=True)
            parent_stat = absolute.parent.lstat()
        except (OSError, RuntimeError) as exc:
            raise AuthorizationGuardUnavailable(
                "authorization guard parent is unavailable"
            ) from exc
        if _is_redirect(parent_stat) or os.path.normcase(os.fspath(parent)) != (
            os.path.normcase(os.fspath(absolute.parent))
        ):
            raise AuthorizationGuardUnavailable(
                "authorization guard parent is redirected"
            )
        self.path = parent / absolute.name

    def _validate_existing_file(self) -> os.stat_result:
        try:
            file_stat = self.path.lstat()
        except OSError as exc:
            raise AuthorizationGuardUnavailable(
                "authorization guard has not been initialized"
            ) from exc
        if _is_redirect(file_stat) or not stat.S_ISREG(file_stat.st_mode):
            raise AuthorizationGuardUnavailable(
                "authorization guard path is unsafe"
            )
        return file_stat

    def ensure_writable(self) -> None:
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            self._validate_existing_file()
        except OSError as exc:
            raise AuthorizationGuardUnavailable(
                "authorization guard cannot be initialized"
            ) from exc
        else:
            os.close(descriptor)
            self._validate_existing_file()

    @contextlib.contextmanager
    def acquire_exclusive(self, *, timeout: float = 0) -> Iterator[None]:
        self.ensure_writable()
        lock = FileLock(
            self.path,
            preserve_lock_file=True,
        )
        try:
            lock.acquire(timeout=timeout)
        except Timeout:
            raise
        except OSError as exc:
            raise AuthorizationGuardUnavailable(
                "authorization guard cannot be acquired"
            ) from exc
        try:
            yield
        finally:
            lock.release()

    @contextlib.contextmanager
    def acquire_readonly(self) -> Iterator[None]:
        expected = self._validate_existing_file()
        if _IS_WINDOWS:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_RDWR | getattr(os, "O_BINARY", 0),
                )
            except OSError as exc:
                raise AuthorizationGuardUnavailable(
                    "authorization guard cannot be opened read-only"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != expected.st_dev
                    or opened.st_ino != expected.st_ino
                ):
                    raise AuthorizationGuardUnavailable(
                        "authorization guard changed while opening"
                    )
                try:
                    acquired = _lock_readonly_descriptor(descriptor)
                except OSError as exc:
                    raise AuthorizationGuardUnavailable(
                        "authorization guard cannot be acquired"
                    ) from exc
                if not acquired:
                    raise AuthorizationGuardUnavailable(
                        "authorization guard is busy"
                    )
                try:
                    yield
                finally:
                    _unlock_readonly_descriptor(descriptor)
            finally:
                os.close(descriptor)
            return
        try:
            import fcntl

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise AuthorizationGuardUnavailable(
                "authorization guard cannot be opened read-only"
            ) from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise AuthorizationGuardUnavailable(
                        "authorization guard is busy"
                    ) from exc
                raise AuthorizationGuardUnavailable(
                    "authorization guard cannot be acquired"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
