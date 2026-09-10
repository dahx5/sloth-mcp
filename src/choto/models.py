from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from choto.config import MAX_ACTIONS_PER_PLAN


class BBox(BaseModel):
    """Axis-aligned bounding box in screen pixels (origin top-left)."""

    x: int
    y: int
    w: int = Field(ge=0)
    h: int = Field(ge=0)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class OcrLine(BaseModel):
    """A single line of recognized text with its location and confidence."""

    text: str
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class ScreenState(BaseModel):
    """A parsed snapshot of the screen: OCR lines plus identity metadata."""

    app_name: str
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    phash: str
    lines: list[OcrLine]


class Region(str, Enum):
    """Which part of the surface a step means, when two elements read alike.

    It is a **preference between equals, not a filter**, and it is measured
    inside the window the step works in rather than across the display. Both
    halves of that sentence are corrections a live run paid for: a plan clicking
    "Previous character" with ``region="left"`` was refused although the button
    was on screen and matched exactly — it sat at x=50% of the *display*, one
    pixel outside a rectangle that has nothing to do with the window — and the
    executor then scrolled the game five notches down and five back looking for
    something it had already found. See
    :meth:`~choto.executor.runner.Executor._resolve_target` for how the
    preference is applied: the target is resolved over the whole surface *and*
    over the named part of it, and the named part wins only when it matched at
    least as well. A region can therefore never hide the only candidate; what it
    can do is decide between two that read the same.

    ``top``/``bottom`` and ``left``/``right`` are halves of the window;
    ``center`` is its middle third on both axes (see
    :func:`~choto.matching.matcher.region_bounds`, which states the geometry
    once for everyone who uses it).
    """

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class Scope(str, Enum):
    """Which surface a target is looked for on: the app's window, or the chrome.

    The two surfaces are disjoint by construction, and a step names exactly one.
    Menus are not part of the window they belong to — the menu bar, the strips
    the system draws over it and any open menu or popover live above every
    window — so "somewhere on screen" would be two different places at once.

    A live failure is why this is stated rather than merged: a plan clicking the
    digits of Calculator's keypad found "5" in the window *and* "5" in the menu
    bar, both exact, and the menu-bar line won on nothing better than list order.
    The click opened a system panel, the step logged a match, and the plan
    reported success over a screen it had littered. Searching one surface at a
    time removes the competition instead of arbitrating it.

    Attributes:
        WINDOW: Only elements inside the window the step works in. The default,
            because that is where a plan's clicks belong; a target that is not
            there is a miss (scrolled for, then escalated), never a fallback
            into the menu bar.
        CHROME: Only the chrome — menu bar, status items, open menus and
            popovers. What a step clicking "File" and then "Save As…" says.
    """

    WINDOW = "window"
    CHROME = "chrome"


class Target(BaseModel):
    """A semantic UI target: the text to match, and where to look for it.

    Attributes:
        text: The visible text to resolve against the screen.
        scope: Which surface to search — the window (default) or the chrome
            around it. See :class:`Scope`; the surfaces are mutually exclusive,
            so a menu item has to be asked for as ``scope="chrome"``.
        region: Which part of the surface the wanted element is in, used to
            choose between two elements that read the same. A preference, never
            a filter — a target that resolves nowhere else still resolves, and
            the report says the region did not decide. Meaningful on both
            scopes: a menu long enough to run down the screen is worth narrowing
            just like a window is. See :class:`Region`.
        window: Substring of the title of the window to work in, matched
            case-insensitively. Omitted, the step works in the *frontmost*
            window of the target application — not in the union of all its
            windows, which is what used to let a plan for one document click
            into another. Which window a step means is a planning decision, so
            it is stated in the plan rather than guessed at by the executor;
            a substring that matches no open window fails the step with the
            titles that are open, instead of acting on the wrong one. Refused
            together with ``scope="chrome"``: chrome belongs to no window, so
            naming one there is a request that cannot be honoured.
        scroll_to_find: Whether the executor may scroll the window and look
            again when this text is not on screen. On by default, because the
            common miss is an item that exists and merely sits below the fold
            (a live run escalated over "Sound" in the System Settings sidebar
            for exactly that reason) and a person would simply have scrolled.
            Turn it off where scrolling is not a harmless look: an endless feed
            no search can exhaust, a list that loads more content as it moves,
            or any place where arriving somewhere else is itself a side effect.
            The switch lives here rather than in the server's configuration
            because the danger is a property of the *list being searched*, which
            only the plan knows; a global setting could only be all-or-nothing.

            Off, and refused if asked for, on ``scope="chrome"``: the search
            scrolls the *window* (its anchor is the window's densest patch of
            text, since a wheel event over the menu bar moves nothing), so
            scrolling to find a menu item would turn the page under the plan
            while looking for something that was never there.
    """

    text: str
    scope: Scope = Scope.WINDOW
    region: Region | None = None
    window: str | None = None
    scroll_to_find: bool = True

    @model_validator(mode="after")
    def _validate_scope(self) -> Target:
        if self.scope is not Scope.CHROME:
            return self
        if self.window is not None:
            raise ValueError(
                "'window' names a window to work in, and scope 'chrome' searches the "
                "menu bar and open menus, which belong to no window; drop one of them."
            )
        if "scroll_to_find" in self.model_fields_set:
            if self.scroll_to_find:
                raise ValueError(
                    "'scroll_to_find' scrolls the window being worked in, which scope "
                    "'chrome' does not search; a menu item is either on screen or the "
                    "menu that holds it has not been opened yet."
                )
            return self
        self.scroll_to_find = False
        return self


class Offset(BaseModel):
    """A pixel displacement from a resolved target's center.

    Exists for the one thing a text-first language cannot say on its own: a spot
    where there is no text. A rubber-band selection starts on empty canvas, a
    resize starts on a window edge — neither has a label to aim at, but both sit
    a predictable distance from one. Anchoring to a target keeps the plan
    semantic: the anchor is resolved against the screen as it is at execution
    time, so a list that scrolled or a window that moved since the plan was
    written still lands the gesture in the right place.

    Attributes:
        dx: Pixels to the right of the target's center (negative = left).
        dy: Pixels below the target's center (negative = up).
    """

    dx: int = 0
    dy: int = 0


class ScreenPoint(BaseModel):
    """A bare pixel position to click, for the control that has no name.

    The emergency exit of a text-first language, and it exists because a live run
    had to invent one: in a game whose buttons are drawn pictures, nothing on the
    dream-selection dialog was in the element listing, and the only way through
    was a zero-length ``drag`` — ``path:[(x,y),(x,y)]`` — which the executor
    performs as a click. A workaround that is needed and works is a missing part
    of the language, not a trick worth keeping secret; and a gesture spelled as a
    drag is also a gesture nobody reading the journey can recognize as a click.

    It is deliberately the *worse* way to write a step, and the executor treats
    it as such (see :meth:`~choto.executor.runner.Executor._click_at`): nothing
    is matched, so nothing can be verified before the button goes down, and the
    journey and the escalation both say the click was aimed by coordinate. Pixels
    are a photograph of where things were when the plan was written — a window
    that moved, a list that scrolled or a display that changed scale invalidates
    them silently, which is exactly what a resolved target cannot do. Reach for
    it only when the control has no text, no label and no annotated icon; name
    the element whenever there is one.

    Attributes:
        x: Horizontal position in captured-frame pixels — the same coordinates
            the element listing reports, not logical points.
        y: Vertical position, in the same pixels.
    """

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class PathPoint(BaseModel):
    """One waypoint of a drag path.

    A point is either an element (``target``, optionally displaced by
    ``offset``) or a raw pixel position (``x``/``y``, in the same coordinate
    space the element listing reports). The two forms are mutually exclusive:
    mixing them would leave it ambiguous which one decides where the gesture
    goes.

    Prefer the anchored form. Raw pixels are a snapshot of where things were
    when the plan was written, while an anchor is resolved against the screen as
    it is when the gesture runs; the executor also cannot check a raw pixel
    against anything, whereas an anchor that no longer exists escalates before a
    single event is sent.
    """

    target: Target | None = None
    offset: Offset | None = None
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_point(self) -> PathPoint:
        has_coordinates = self.x is not None or self.y is not None
        if self.target is not None and has_coordinates:
            raise ValueError(
                "A drag path point is either 'target' (optionally with 'offset') "
                "or explicit 'x'/'y' pixels, not both."
            )
        if self.target is None and not has_coordinates:
            raise ValueError(
                "A drag path point needs either 'target' (optionally with 'offset') "
                "or both 'x' and 'y' in screen pixels."
            )
        if has_coordinates and (self.x is None or self.y is None):
            raise ValueError("A drag path point given by coordinates needs both 'x' and 'y'.")
        if self.offset is not None and self.target is None:
            raise ValueError(
                "'offset' displaces a resolved target, so it requires 'target'; "
                "to name a bare position use 'x'/'y'."
            )
        return self


class ServoReadout(BaseModel):
    """The instrument a drag steers by, and the reading it is steering to.

    Direct manipulation is continuous, and the plan that drives it cannot know
    the pixel arithmetic: how many pixels a timeline is per second depends on the
    zoom, a slider snaps to its detents, and a window that moved invalidates the
    calibration a screenshot was measured on. What the application *does* provide
    is its own answer — the timecode under the playhead, the duration bubble that
    follows a trim, the number beside a slider. So the honest primitive is not
    "drag 240 pixels" but "drag until the readout says 00:12:00": move a little,
    read the instrument, correct. The closed loop forgives snapping, stickiness
    and a calibration nobody took.

    The gesture stays one gesture — the button goes down once and is released
    once, when the reading arrives (or when the executor gives up) — so
    ``modifiers`` remain meaningful and are deliberately **not** refused here: an
    alt-drag that copies and a shift-drag that constrains an axis are steered by
    the same feedback as a plain one, and the modifier changes what the
    application does with the gesture, not how the loop watches it.

    Attributes:
        watch: Where the instrument is. Resolved by the ordinary matcher *before
            the button goes down*, exactly like every waypoint of the path, and
            the rectangle it matched is what gets watched for the rest of the
            gesture — grown by a small margin, since a reading gains and loses
            characters as it counts ("9" becoming "10"). It must name the reading
            itself, not a label beside it: the executor reads the text inside
            that rectangle and no further, so a target on the word "Duration"
            watches the word "Duration" change to nothing. A readout that only
            appears once the drag is underway cannot be named here at all,
            because there is nothing to resolve before the press.
        value: The reading to stop at, written the way the application writes it
            — ``"00:12:00"``, ``"-6 dB"``, ``"50%"``, ``"1 250"``. It is parsed
            by the same readers the ruler calibration uses (timecode, percent,
            decibel, plain number in any common locale), and the reader that
            understands *this* text is the one every reading is then parsed with:
            a value given as ``"12"`` will not be satisfied by a screen showing
            ``"00:00:12"``, because those are two different notations and
            guessing between them is how a plan lands somewhere else.
        tolerance: How far the reading may sit from :attr:`value`, in the units
            :attr:`value` is written in — seconds for a timecode, percent for a
            percentage, dB for a level. Zero (the default) asks for the value
            exactly as parsed, which is the right ask for anything quantised: a
            timeline that only stops on whole frames either shows ``00:12:00`` or
            it does not. Widen it for a continuous control whose exact number is
            unreachable — a volume slider that skips from 49 to 51 — since a
            search that cannot land on the value would otherwise spend its whole
            budget proving it.
    """

    watch: Target
    value: str = Field(min_length=1)
    tolerance: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_watch(self) -> ServoReadout:
        if "scroll_to_find" in self.watch.model_fields_set and self.watch.scroll_to_find:
            raise ValueError(
                "'scroll_to_find' cannot be asked for on the readout a drag watches: "
                "scrolling would move the readout, the control and the press point "
                "together, and the gesture is aimed at all three at once."
            )
        self.watch = self.watch.model_copy(update={"scroll_to_find": False})
        return self


class Expect(BaseModel):
    """Post-condition for a step. At least one field must be set.

    Attributes:
        appears: Text that must be present on screen after the step.
        disappears: Text that must be gone after the step. Only correct for text
            that truly leaves the screen — a text field that shows its
            placeholder again once emptied never satisfies it.
        appears_count_increases: Text whose number of matching on-screen lines
            must be strictly greater than it was before the step. The right
            check for "one more of these appeared" (a sent message joining a
            feed) where the same text is also still visible elsewhere.
        screen_changes: Whether the screen had to visibly change. Judged by two
            independent signals — the perceptual hash (a whole-screen summary,
            which catches navigation) and the share of changed pixels (which
            catches local edits the hash is structurally blind to, such as a
            window opening over a blank document or a checkbox being ticked).
            ``True`` needs either to fire; ``False`` needs both to stay silent.
    """

    appears: str | None = None
    disappears: str | None = None
    appears_count_increases: str | None = None
    screen_changes: bool | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> Expect:
        conditions = type(self).model_fields
        if any(getattr(self, name) is not None for name in conditions):
            return self
        raise ValueError(f"Expect requires at least one of: {', '.join(conditions)}.")


class Action(str, Enum):
    """The atomic actions the executor can perform.

    Most of them *do* something. Two of them bring something back:
    :attr:`READ` and :attr:`READ_CLIPBOARD` are the plan's way of harvesting
    text, and their product is the report's ``extracted`` block rather than a
    change on screen.
    """

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    WAIT = "wait"
    FOCUS_APP = "focus_app"
    READ = "read"
    READ_CLIPBOARD = "read_clipboard"


_TARGET_ACTIONS = frozenset({Action.CLICK, Action.DOUBLE_CLICK, Action.RIGHT_CLICK})

_READ_ONLY_FIELDS = ("from_text", "to_text", "scroll")

_MODIFIER_ACTIONS = _TARGET_ACTIONS | {Action.DRAG}

_MIN_PATH_POINTS = 2

_SERVO_PATH_POINTS = 2


class Step(BaseModel):
    """A single instruction in a plan.

    Field requirements are enforced per-action by :meth:`_validate_action`:

    - click / double_click / right_click require ``target`` **or** ``at``, and
      refuse both at once
    - drag requires ``path`` of at least two points, and exactly two when it is
      steered by ``until_readout``
    - type requires ``text``
    - hotkey requires ``keys``
    - scroll requires ``amount`` (sign = direction, positive = down)
    - focus_app requires ``app_name``
    - wait requires ``expect`` or a positive ``timeout_ms``
    - read takes an optional ``target`` with an empty ``text`` (it says where to
      read, not what to aim at) and the options ``from_text`` / ``to_text`` /
      ``scroll``
    - read_clipboard takes no ``target``
    - ``modifiers`` are only accepted by the actions that can hold them
    - ``skip_if_absent`` requires ``target``

    Attributes:
        at: For a click — the pixel to press, instead of an element to resolve.
            The way to reach a control that no text names (see
            :class:`ScreenPoint`); mutually exclusive with ``target``, because a
            step that named both would leave it undecided which of the two
            decides where the click lands. The point is checked against the
            window the step works in and against what is drawn over it, but
            nothing about it can be *matched*, so the journey marks the click as
            aimed by coordinate and the report repeats it.
        from_text: For ``read`` — start the reading at the line this text
            resolves to, that line included. Resolved by the ordinary matcher,
            on the same surface the step reads.
        to_text: For ``read`` — stop the reading at the line this text resolves
            to, that line included.
        scroll: For ``read`` — read the document to its end instead of reading
            the screenful in front of the plan: the window is scrolled and
            re-read until it stops yielding new content, the pieces are stitched
            on their overlap, and the scroll position is put back afterwards.
            Off by default, because most reads are of a pane that is fully
            visible and scrolling is never free — it costs a parse per screenful
            and moves a list the user is looking at.
        until_readout: For ``drag`` — the application's own instrument to steer
            the gesture by, instead of releasing at the end of the path. See
            :class:`ServoReadout`: the button goes down on the first waypoint,
            the pointer is pushed toward the second one and the reading is
            checked after every push, until it says what the plan asked for.
        skip_if_absent: Whether a target that does not resolve makes this step a
            no-op instead of an escalation. For the known noise of a real
            desktop — a cookie banner, a "what's new" sheet, an update prompt —
            which may or may not be there when the plan arrives. Both branches
            have to converge on the same screen for this to be correct, which is
            why it is stated per step by the plan rather than inferred: only the
            plan knows that dismissing the banner and never seeing it lead to
            the same place. A skipped step's ``expect`` is not checked; it never
            happened.
        timeout_ms: The step's whole budget, not one phase of it. Winning the
            front back, scrolling a target into view, following a target that is
            still moving, waiting for the picture to settle and checking
            ``expect`` all draw on this one clock, and the step ends when it runs
            out — so the number written here is the longest the machine can hold
            the user's screen for this step. A bare ``wait`` (one with no
            ``expect``) spends it on purpose: it watches the window for this long
            and comes back sooner only once something has moved and stopped
            again, which is how a plan gives a slow export the time to finish.
    """

    action: Action
    target: Target | None = None
    at: ScreenPoint | None = None
    path: list[PathPoint] | None = None
    modifiers: list[str] | None = None
    text: str | None = None
    keys: list[str] | None = None
    amount: int | None = None
    app_name: str | None = None
    from_text: str | None = None
    to_text: str | None = None
    scroll: bool = False
    until_readout: ServoReadout | None = None
    skip_if_absent: bool = False
    expect: Expect | None = None
    timeout_ms: int = Field(default=5000, ge=1)

    @model_validator(mode="after")
    def _validate_action(self) -> Step:
        self._validate_aim()
        if self.action is Action.DRAG:
            self._validate_drag()
        elif self.until_readout is not None:
            raise ValueError(
                "'until_readout' steers a held gesture by what the application shows while "
                f"it is held, which only 'drag' does, not {self.action.value!r}."
            )
        if self.modifiers and self.action not in _MODIFIER_ACTIONS:
            allowed = ", ".join(sorted(action.value for action in _MODIFIER_ACTIONS))
            raise ValueError(
                f"'modifiers' are only held by {allowed}, not by {self.action.value!r}."
            )
        if self.action is Action.TYPE and self.text is None:
            raise ValueError("Action 'type' requires 'text'.")
        if self.action is Action.HOTKEY and not self.keys:
            raise ValueError("Action 'hotkey' requires a non-empty 'keys'.")
        if self.action is Action.SCROLL and self.amount is None:
            raise ValueError("Action 'scroll' requires 'amount'.")
        if self.action is Action.FOCUS_APP and not self.app_name:
            raise ValueError("Action 'focus_app' requires 'app_name'.")
        if self.action is Action.READ_CLIPBOARD and self.target is not None:
            raise ValueError(
                "Action 'read_clipboard' reads the system clipboard, which belongs to no "
                "window, so it takes no 'target'; use 'read' to read a window."
            )
        self._validate_reading()
        return self

    def _validate_aim(self) -> None:
        if self.action in _TARGET_ACTIONS:
            if self.target is None and self.at is None:
                raise ValueError(
                    f"Action {self.action.value!r} requires 'target' (an element to resolve "
                    "by its text) or 'at' (a bare pixel, for a control no text names)."
                )
            if self.target is not None and self.at is not None:
                raise ValueError(
                    f"Action {self.action.value!r} takes 'target' or 'at', not both: the first "
                    "is resolved against the screen as it is now, the second is a position "
                    "written down when the plan was, and only one of them can decide where "
                    "the click lands."
                )
            return
        if self.at is not None:
            raise ValueError(
                "'at' is the pixel a click presses when no text names the control, and only "
                f"click/double_click/right_click press one, not {self.action.value!r}."
            )

    def _validate_reading(self) -> None:
        if self.action is not Action.READ:
            misplaced = [name for name in _READ_ONLY_FIELDS if getattr(self, name)]
            if misplaced:
                raise ValueError(
                    f"{', '.join(repr(name) for name in misplaced)} describe how to read text "
                    f"off the screen and only apply to action 'read', not to "
                    f"{self.action.value!r}."
                )
        if self.skip_if_absent and self.target is None:
            raise ValueError(
                "'skip_if_absent' skips the step when its target cannot be resolved, so it "
                f"requires 'target'; action {self.action.value!r} has none."
            )
        if self.action is not Action.READ or self.target is None:
            return
        if self.target.text:
            raise ValueError(
                "A 'read' step has no element to aim at: 'target' on a read only says "
                "*where* to read (window / scope / region), so its 'text' must be empty. "
                "To read from one piece of text to another, use 'from_text' / 'to_text'."
            )
        if self.scroll and self.target.scope is Scope.CHROME:
            raise ValueError(
                "'scroll' reads a document to its end by scrolling the window, and scope "
                "'chrome' reads the menu bar and open menus, which do not scroll; a menu is "
                "either on screen or the menu that holds it has not been opened yet."
            )

    def _validate_drag(self) -> None:
        if not self.path or len(self.path) < _MIN_PATH_POINTS:
            given = len(self.path) if self.path else 0
            raise ValueError(
                f"Action 'drag' requires 'path' with at least {_MIN_PATH_POINTS} points "
                f"(where the button goes down and where it is released), got {given}."
            )
        windows = {
            point.target.window
            for point in self.path
            if point.target is not None and point.target.window
        }
        if len(windows) > 1:
            listed = ", ".join(repr(window) for window in sorted(windows))
            raise ValueError(
                "A drag happens inside one window: every path point that sets "
                f"'target.window' must name the same window, got {listed}."
            )
        if self.until_readout is not None and len(self.path) != _SERVO_PATH_POINTS:
            raise ValueError(
                f"A drag steered by 'until_readout' takes exactly {_SERVO_PATH_POINTS} path "
                f"points — where the button goes down and which way to push — got "
                f"{len(self.path)}; the readout decides where the gesture ends."
            )


class Plan(BaseModel):
    """An ordered batch of steps (1..MAX_ACTIONS_PER_PLAN).

    The ceiling counts *actions*, which is why it is the same number the executor
    budgets its runs with: a plan of fifty steps leaves a target search nothing
    to scroll with, and a plan longer than the configured budget
    (``Settings.max_actions_per_plan``) is refused before anything is clicked
    rather than executed until the budget runs out somewhere in the middle.
    """

    steps: list[Step] = Field(min_length=1, max_length=MAX_ACTIONS_PER_PLAN)


class RunStatus(str, Enum):
    """Terminal (and running) states of an execution run."""

    RUNNING = "running"
    SUCCESS = "success"
    ESCALATED = "escalated"
    ABORTED = "aborted"


class ElementKind(str, Enum):
    """How an element's text came to be known.

    A window's element set used to be exactly what OCR read, so every element
    carried its own label by construction. Icon detection breaks that: the
    detector finds *where* a clickable glyph sits and has no idea what it means,
    so an icon element is born with an empty text and is named later, once, by
    the labelling pass — after which it is an ordinary clickable element like
    any other.

    The kind is what keeps the two apart while that is still true. A text
    element with no text is debris; an icon element with no text is a known
    place awaiting its name, and it is the row the labelling pass writes back
    to (see :class:`~choto.graph.db_models.IconGlyph`).

    Attributes:
        TEXT: Read by OCR. The default, and what every element written before
            icons existed is.
        ICON: Found by the icon detector; its ``icon_phash`` says which glyph
            was drawn there.
    """

    TEXT = "text"
    ICON = "icon"


class IconLabelSource(str, Enum):
    """Where a glyph's label came from, or that it has none yet.

    Recorded rather than inferred from the label's presence, because the two
    sources are not equally trustworthy and a later pass has to be able to tell
    them apart: a tooltip is the application's own word for the control, while
    an LLM reading a contact sheet is a guess — a good one, but a guess.

    Attributes:
        UNLABELED: The glyph has been seen but never named. Pairs with an empty
            ``label``; the two always change together.
        LLM: Written by the labelling pass from a sheet of glyph crops.
        TOOLTIP: Read off the application's own tooltip for the control.
    """

    UNLABELED = ""
    LLM = "llm"
    TOOLTIP = "tooltip"


class SeenElement(BaseModel):
    """A text element kept in a run's screen ledger, with its pixel center."""

    text: str
    x: int
    y: int


class SeenScreen(BaseModel):
    """A distinct window a run observed, and where in the plan it was seen.

    The executor keeps one of these per window it read during a run — including
    the windows of steps that failed — so an escalation can tell Claude what
    passed by earlier in the plan, not only the window it ended on. A menu whose
    item was labelled differently than the plan assumed is exactly the kind of
    thing that is already known by then and must not be thrown away.

    Attributes:
        phash: Perceptual hash of the window crop; also the ledger's dedup key.
        app_name: Application that owned the window when it was read.
        window_title: Title of that window, or ``""`` when it has none. Carried
            because it is the handle a corrected plan uses: ``target.window``
            names a window by a substring of exactly this text.
        scoped: Whether a window was actually located for this sighting. ``False``
            means the window server could not say where the app was and the read
            covered the whole display, so the elements below belong to no single
            window and possibly to several applications. Such a sighting is
            deliberately not remembered in the interface graph, and rendering it
            as "some untitled window of this app" would assert exactly what the
            run failed to establish.
        steps: Plan step numbers at which this window was observed, in order
            (``0`` means "before the first step ran"). A window revisited later
            is listed once, with every step it appeared at.
        action: Action of the step that first showed this window.
        elements: A bounded, filtered selection of the window's text elements —
            not the full listing; see :attr:`element_count`.
        element_count: How many text lines the read actually had, so a
            truncated :attr:`elements` can be reported honestly.
    """

    phash: str
    app_name: str
    window_title: str = ""
    scoped: bool = True
    steps: list[int]
    action: str
    elements: list[SeenElement]
    element_count: int = Field(ge=0)


class ExtractedSource(str, Enum):
    """Where a piece of harvested text was taken from.

    Attributes:
        SCREEN: Read off the screen by a ``read`` step, laid out as it looked.
        CLIPBOARD: Taken from the system clipboard by a ``read_clipboard`` step.
    """

    SCREEN = "screen"
    CLIPBOARD = "clipboard"


class ExtractedText(BaseModel):
    """Text a plan went and fetched, kept whole.

    This is the one thing a report carries that is deliberately **not** capped.
    Every other rendering in the reply is budgeted — the screenshot, the element
    listing, the window ledger — because they are context around a decision, and
    an unbounded one would undo the token economy Choto exists for. What a
    ``read`` step brings back is not context: it is the answer the plan was
    written to obtain, and a truncated answer is a wrong one that looks right.

    Attributes:
        step_index: 1-based number of the step that fetched it.
        source: Screen or clipboard; see :class:`ExtractedSource`.
        window_title: Title of the window the step was working in, ``""`` when
            it has none or when no window could be located. Recorded for a
            clipboard read too — not because the clipboard belongs to that
            window, but because it is where the plan was standing when it
            copied, which is what makes the text identifiable later.
        text: The text itself, verbatim.
    """

    step_index: int = Field(ge=1)
    source: ExtractedSource
    window_title: str = ""
    text: str


class ExecutionReport(BaseModel):
    """Result of executing a plan, returned to Claude for (re)planning.

    ``elements`` describes the screen the run ended on; ``seen_screens`` is the
    ledger of every distinct screen the run passed through, which the MCP layer
    renders (under strict size caps) only when the run needs Claude's help.
    ``extracted`` is what the run was sent to fetch, and travels whatever the
    status: a plan that read three panes and then failed on the fourth still did
    three quarters of its work, and throwing that away would make the corrected
    plan repeat it.
    """

    status: RunStatus
    completed_steps: list[int]
    failed_step: int | None = None
    reason: str | None = None
    journey: list[str]
    screenshot_png: bytes | None = None
    elements: list[str]
    seen_screens: list[SeenScreen] = Field(default_factory=list)
    extracted: list[ExtractedText] = Field(default_factory=list)
