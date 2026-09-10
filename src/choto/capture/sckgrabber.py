from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import cv2
import numpy as np

from choto.capture.framing import clamp_box
from choto.capture.grabber import CaptureError, has_screen_recording_permission
from choto.capture.monitor import MonitorInfo
from choto.capture.source import Frame
from choto.log import get_logger
from choto.models import BBox

_log = get_logger(__name__)

DEFAULT_FRAME_RATE_HZ = 60

_QUEUE_DEPTH = 4

_CONTENT_TIMEOUT_S = 10.0
_START_TIMEOUT_S = 10.0
_STOP_TIMEOUT_S = 5.0
_REFIT_TIMEOUT_S = 5.0
_FIRST_FRAME_TIMEOUT_S = 5.0

_REFIT_ATTEMPTS = 2

_STATUS_COMPLETE = 0
_STATUS_ABSENT = -1

_COVERAGE_TOLERANCE_PX = 1.0

_NO_CONTENT_SIZE = (0.0, 0.0)
_NO_SCALE = 0.0


def _now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass(frozen=True)
class _Held:
    sample_buffer: object
    pixel_buffer: object
    captured_at_ms: float
    covers_display: bool


@dataclass(frozen=True)
class _Mark:
    at_ms: float
    frames: int


class _FrameHolder:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._held: _Held | None = None
        self._error = ""
        self._frames = 0

    def offer(self, held: _Held) -> None:
        with self._cond:
            self._held = held
            self._frames += 1
            self._cond.notify_all()

    def fail(self, message: str) -> None:
        with self._cond:
            self._error = message
            self._cond.notify_all()

    def wait(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._held is None and not self._error:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(remaining)
            return True

    def mark(self) -> _Mark:
        with self._cond:
            return _Mark(at_ms=_now_ms(), frames=self._frames)

    def wait_after(self, mark: _Mark, *, frames: int, timeout_s: float) -> _Held | None:
        if frames < 1:
            raise ValueError(f"A frame barrier must wait for at least one frame, got {frames}.")
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                if self._error:
                    return None
                held = self._held
                if (
                    self._frames - mark.frames >= frames
                    and held is not None
                    and held.captured_at_ms > mark.at_ms
                ):
                    return held
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(remaining)

    def take(self) -> tuple[_Held | None, str]:
        with self._cond:
            return self._held, self._error

    def release(self) -> None:
        with self._cond:
            self._held = None

    @property
    def frames(self) -> int:
        with self._cond:
            return self._frames


def covers_display(
    content_width_pt: float,
    content_height_pt: float,
    scale_factor: float,
    width_px: int,
    height_px: int,
) -> bool:
    if scale_factor <= 0.0 or content_width_pt <= 0.0 or content_height_pt <= 0.0:
        return False
    covered_w = content_width_pt * scale_factor
    covered_h = content_height_pt * scale_factor
    return (
        abs(covered_w - width_px) <= _COVERAGE_TOLERANCE_PX
        and abs(covered_h - height_px) <= _COVERAGE_TOLERANCE_PX
    )


def _content_size(rect: object) -> tuple[float, float]:
    import Quartz

    if rect is None:
        return _NO_CONTENT_SIZE
    ok, content = Quartz.CGRectMakeWithDictionaryRepresentation(rect, None)
    if not ok:
        return _NO_CONTENT_SIZE
    return (float(content.size.width), float(content.size.height))


_output_class_cache: type | None = None
_output_class_lock = threading.Lock()


def _output_class() -> type:
    global _output_class_cache
    with _output_class_lock:
        if _output_class_cache is None:
            _output_class_cache = _build_output_class()
        return _output_class_cache


def _build_output_class() -> type:
    import CoreMedia
    import objc
    import Quartz
    import ScreenCaptureKit as SCK
    from Foundation import NSObject

    status_key = SCK.SCStreamFrameInfoStatus
    content_key = SCK.SCStreamFrameInfoContentRect
    scale_key = SCK.SCStreamFrameInfoScaleFactor

    class _StreamOutput(  # type: ignore[misc]
        NSObject,
        protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")],
    ):
        def initWithHolder_(self, holder: _FrameHolder) -> object:  # noqa: N802
            this = objc.super(_StreamOutput, self).init()
            if this is None:
                return None
            this._holder = holder
            return this

        def stream_didOutputSampleBuffer_ofType_(  # noqa: N802
            self, stream: object, sample_buffer: object, output_type: object
        ) -> None:
            try:
                attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(
                    sample_buffer, False
                )
                if not attachments:
                    return
                info = attachments[0]
                if int(info.get(status_key, _STATUS_ABSENT)) != _STATUS_COMPLETE:
                    return
                pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
                if pixel_buffer is None:
                    return
                content_width_pt, content_height_pt = _content_size(info.get(content_key))
                covered = covers_display(
                    content_width_pt,
                    content_height_pt,
                    float(info.get(scale_key, _NO_SCALE)),
                    int(Quartz.CVPixelBufferGetWidth(pixel_buffer)),
                    int(Quartz.CVPixelBufferGetHeight(pixel_buffer)),
                )
                self._holder.offer(
                    _Held(
                        sample_buffer=sample_buffer,
                        pixel_buffer=pixel_buffer,
                        captured_at_ms=_now_ms(),
                        covers_display=covered,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - see the class docstring.
                _log.error("sckgrabber.frame_failed", error=str(exc))

        def stream_didStopWithError_(self, stream: object, error: object) -> None:  # noqa: N802
            try:
                message = str(error.localizedDescription()) if error is not None else "unknown"
            except Exception:  # noqa: BLE001 - a broken error object is still a stop.
                message = "unknown"
            _log.error("sckgrabber.stream_stopped", error=message)
            self._holder.fail(f"The capture stream stopped: {message}")

    return _StreamOutput


def _rgb_from_pixel_buffer(pixel_buffer: object, box: BBox | None) -> np.ndarray:
    import Quartz

    flags = Quartz.kCVPixelBufferLock_ReadOnly
    status = Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, flags)
    if status != 0:
        raise CaptureError(f"Could not lock the captured frame for reading (CVReturn {status}).")
    try:
        width = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
        height = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))
        stride = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
        base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        if base is None:
            raise CaptureError("The captured frame has no readable pixels.")
        bgra = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8)
        bgra = bgra.reshape(height, stride // 4, 4)[:, :width, :]
        if box is not None:
            bgra = bgra[box.y : box.y + box.h, box.x : box.x + box.w, :]
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, flags)


def _shareable_content() -> object:
    import ScreenCaptureKit as SCK

    answer: dict[str, object] = {}
    done = threading.Event()

    def handler(content: object, error: object) -> None:
        answer["content"] = content
        answer["error"] = error
        done.set()

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    if not done.wait(_CONTENT_TIMEOUT_S):
        raise CaptureError(
            f"ScreenCaptureKit did not answer what is capturable within {_CONTENT_TIMEOUT_S:.0f} s."
        )
    error = answer.get("error")
    if error is not None:
        raise CaptureError(f"ScreenCaptureKit refused to list capturable content: {error}.")
    content = answer.get("content")
    if content is None:
        raise CaptureError("ScreenCaptureKit listed no capturable content.")
    return content


def _completed(
    call: Callable[[Callable[[object], None]], None],
    *,
    timeout_s: float,
    timed_out: str,
    refused: str,
) -> None:
    answer: dict[str, object] = {}
    done = threading.Event()

    def handler(error: object) -> None:
        answer["error"] = error
        done.set()

    call(handler)
    if not done.wait(timeout_s):
        raise CaptureError(timed_out)
    error = answer.get("error")
    if error is not None:
        raise CaptureError(f"{refused}: {error}.")


def _main_display(content: object) -> object:
    import Quartz

    main_id = int(Quartz.CGMainDisplayID())
    displays = list(content.displays())
    for display in displays:
        if int(display.displayID()) == main_id:
            return display
    raise CaptureError(
        f"The main display (id {main_id}) is not among the "
        f"{len(displays)} display(s) ScreenCaptureKit offers."
    )


class SckScreenGrabber:
    def __init__(
        self,
        *,
        frame_rate_hz: int = DEFAULT_FRAME_RATE_HZ,
    ) -> None:
        if frame_rate_hz <= 0:
            raise ValueError(f"Frame rate must be positive, got {frame_rate_hz}.")
        if not has_screen_recording_permission():
            raise CaptureError(
                "Screen Recording permission is not granted to this process, so "
                "ScreenCaptureKit has nothing to stream. Grant it in System "
                "Settings > Privacy & Security > Screen Recording."
            )

        import ScreenCaptureKit as SCK

        self._frame_rate_hz = frame_rate_hz
        self._closed = False
        self._holder = _FrameHolder()
        self._refit_lock = threading.Lock()

        self._filter = self._display_filter()
        width, height = self._output_size(self._filter)
        config = self._configuration(width, height)

        self._output = _output_class().alloc().initWithHolder_(self._holder)
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            self._filter, config, self._output
        )
        added, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._output, SCK.SCStreamOutputTypeScreen, None, None
        )
        if not added:
            raise CaptureError(f"ScreenCaptureKit refused the frame consumer: {error}.")
        try:
            self._start()
        except CaptureError:
            self.close()
            raise
        _log.info(
            "sckgrabber.started",
            width=width,
            height=height,
            frame_rate_hz=frame_rate_hz,
        )

    def _display_filter(self) -> object:
        import ScreenCaptureKit as SCK

        content = _shareable_content()
        display = _main_display(content)
        return SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

    def _output_size(self, content_filter: object) -> tuple[int, int]:
        rect = content_filter.contentRect()
        return (
            int(round(float(rect.size.width))),
            int(round(float(rect.size.height))),
        )

    def _configuration(self, width: int, height: int) -> object:
        import CoreMedia
        import Quartz
        import ScreenCaptureKit as SCK

        config = SCK.SCStreamConfiguration.alloc().init()
        config.setWidth_(width)
        config.setHeight_(height)
        config.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
        config.setShowsCursor_(False)
        config.setQueueDepth_(_QUEUE_DEPTH)
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self._frame_rate_hz))
        return config

    def _start(self) -> None:
        _completed(
            self._stream.startCaptureWithCompletionHandler_,
            timeout_s=_START_TIMEOUT_S,
            timed_out=f"The capture stream did not start within {_START_TIMEOUT_S:.0f} s.",
            refused="The capture stream refused to start",
        )

    def __enter__(self) -> SckScreenGrabber:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def broken(self) -> str:
        _, error = self._holder.take()
        return error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        done = threading.Event()

        def handler(error: object) -> None:
            if error is not None:
                _log.warning("sckgrabber.stop_failed", error=str(error))
            done.set()

        try:
            self._stream.stopCaptureWithCompletionHandler_(handler)
            if not done.wait(_STOP_TIMEOUT_S):
                _log.warning("sckgrabber.stop_timeout", seconds=_STOP_TIMEOUT_S)
        finally:
            self._holder.release()
            _log.info("sckgrabber.stopped", frames=self._holder.frames)

    def _newest(self) -> _Held:
        if self._closed:
            raise CaptureError("This grabber is closed; its capture stream has been stopped.")
        if not self._holder.wait(_FIRST_FRAME_TIMEOUT_S):
            raise CaptureError(
                f"ScreenCaptureKit delivered no frame within {_FIRST_FRAME_TIMEOUT_S:.0f} s "
                f"of the stream starting."
            )
        held, error = self._holder.take()
        if error:
            raise CaptureError(error)
        if held is None:
            raise CaptureError("The capture stream reported a frame and then had none.")
        if not held.covers_display:
            return self._refit()
        return held

    def _covering(self) -> _Held | None:
        held, error = self._holder.take()
        if error:
            raise CaptureError(error)
        return held if held is not None and held.covers_display else None

    def _refit(self) -> _Held:
        with self._refit_lock:
            for attempt in range(1, _REFIT_ATTEMPTS + 1):
                covering = self._covering()
                if covering is not None:
                    return covering
                self._refit_once(attempt)
            covering = self._covering()
            if covering is not None:
                return covering
        raise CaptureError(
            f"The captured frame no longer covers the display, and {_REFIT_ATTEMPTS} "
            "attempts to rebuild the capture stream around the new geometry all "
            "found it changed again; the display is still being resized."
        )

    def _refit_once(self, attempt: int) -> None:
        new_filter = self._display_filter()
        width, height = self._output_size(new_filter)
        mark = self._holder.mark()
        self._apply(
            lambda handler: self._stream.updateContentFilter_completionHandler_(
                new_filter, handler
            ),
            "content filter",
        )
        self._apply(
            lambda handler: self._stream.updateConfiguration_completionHandler_(
                self._configuration(width, height), handler
            ),
            "configuration",
        )
        self._filter = new_filter
        fresh = self._holder.wait_after(mark, frames=_QUEUE_DEPTH, timeout_s=_FIRST_FRAME_TIMEOUT_S)
        _, error = self._holder.take()
        if error:
            raise CaptureError(error)
        if fresh is None:
            raise CaptureError(
                f"The capture stream delivered no frame within {_FIRST_FRAME_TIMEOUT_S:.0f} s "
                f"of being rebuilt for a {width}x{height} display."
            )
        _log.info(
            "sckgrabber.refitted",
            width=width,
            height=height,
            attempt=attempt,
            covers_display=fresh.covers_display,
        )

    def _apply(self, update: Callable[[Callable[[object], None]], None], what: str) -> None:
        _completed(
            update,
            timeout_s=_REFIT_TIMEOUT_S,
            timed_out=(
                f"The capture stream did not accept a new {what} within {_REFIT_TIMEOUT_S:.0f} s."
            ),
            refused=f"The capture stream refused the new {what}",
        )

    def _size(self, pixel_buffer: object) -> tuple[int, int]:
        import Quartz

        return (
            int(Quartz.CVPixelBufferGetWidth(pixel_buffer)),
            int(Quartz.CVPixelBufferGetHeight(pixel_buffer)),
        )

    def grab(self) -> np.ndarray:
        return self.latest().image

    def grab_region(self, box: BBox) -> np.ndarray:
        return self.region(box).image

    def display_points(self) -> tuple[int, int]:
        import Quartz

        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(round(bounds.size.width)), int(round(bounds.size.height))

    def monitor_info(self) -> MonitorInfo:
        held = self._newest()
        width_px, height_px = self._size(held.pixel_buffer)
        width_pt, height_pt = self.display_points()
        scale = width_px / width_pt if width_pt else 1.0
        return MonitorInfo(
            width_px=width_px,
            height_px=height_px,
            width_pt=width_pt,
            height_pt=height_pt,
            scale=scale,
        )

    def latest(self) -> Frame:
        held = self._newest()
        image = _rgb_from_pixel_buffer(held.pixel_buffer, None)
        return Frame(
            image=image,
            box=BBox(x=0, y=0, w=int(image.shape[1]), h=int(image.shape[0])),
            captured_at_ms=held.captured_at_ms,
            age_ms=max(0.0, _now_ms() - held.captured_at_ms),
        )

    def region(self, box: BBox) -> Frame:
        held = self._newest()
        width, height = self._size(held.pixel_buffer)
        clamped = clamp_box(box, width, height)
        if clamped.w == 0 or clamped.h == 0:
            raise ValueError(
                f"Region {box.x},{box.y},{box.w},{box.h} lies outside the {width}x{height} display."
            )
        image = _rgb_from_pixel_buffer(held.pixel_buffer, clamped)
        return Frame(
            image=image,
            box=clamped,
            captured_at_ms=held.captured_at_ms,
            age_ms=max(0.0, _now_ms() - held.captured_at_ms),
        )

    def geometry(self) -> MonitorInfo:
        return self.monitor_info()
