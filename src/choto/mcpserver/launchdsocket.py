from __future__ import annotations

import ctypes
import errno
import os

from choto.log import get_logger

_log = get_logger(__name__)

_LIBSYSTEM_PATH = "/usr/lib/libSystem.B.dylib"

_ACTIVATION_ERRORS = {
    errno.ENOENT: (
        "the launchd job declares no socket by that name — the agent's plist has "
        "no matching Sockets entry (rewrite it with `choto service install`)"
    ),
    errno.ESRCH: (
        "this process is not a launchd job, so there is no socket to inherit "
        "(that is the normal answer when the daemon was started by hand)"
    ),
    errno.EALREADY: (
        "the descriptors for that socket were already collected by this process; "
        "launchd hands them over exactly once"
    ),
}


class SocketActivationError(RuntimeError): ...


def _libsystem() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL(_LIBSYSTEM_PATH, use_errno=True)
    except OSError as exc:
        raise SocketActivationError(f"cannot load {_LIBSYSTEM_PATH}: {exc}") from exc

    library.launch_activate_socket.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.launch_activate_socket.restype = ctypes.c_int
    library.free.argtypes = [ctypes.c_void_p]
    library.free.restype = None
    return library


def activate_socket(name: str) -> int:
    library = _libsystem()
    descriptors = ctypes.POINTER(ctypes.c_int)()
    count = ctypes.c_size_t(0)

    status = library.launch_activate_socket(
        name.encode("utf-8"), ctypes.byref(descriptors), ctypes.byref(count)
    )
    if status != 0:
        detail = _ACTIVATION_ERRORS.get(status, os.strerror(status))
        raise SocketActivationError(f"launchd did not hand over the socket {name!r}: {detail}")

    try:
        inherited = [descriptors[index] for index in range(count.value)]
    finally:
        library.free(descriptors)

    if len(inherited) != 1:
        for descriptor in inherited:
            os.close(descriptor)
        raise SocketActivationError(
            f"launchd handed over {len(inherited)} descriptors for the socket {name!r}; "
            "the daemon serves exactly one endpoint"
        )

    _log.info("socket.activated", name=name, fd=inherited[0])
    return inherited[0]
