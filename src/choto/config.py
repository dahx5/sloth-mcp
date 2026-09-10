from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from choto import userpaths

MAX_ACTIONS_PER_PLAN = 50

STABLE_POLL_INTERVAL_MS = 50

OCR_ENGINE_APPLE_VISION = "apple_vision"

PLATFORM_AUTO = "auto"
PLATFORM_MACOS = "macos"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHOTO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: Path = Field(
        default_factory=userpaths.default_db_path,
        description=(
            "Filesystem path to the SQLite database. The default is 'data/choto.db' "
            "relative to the project root, which the entry point anchors, so it never "
            "depends on the working directory a supervisor hands the daemon. See "
            "choto.userpaths for why it lives in the checkout."
        ),
    )

    max_mcp_sessions: int = Field(
        default=8,
        ge=1,
        description=(
            "How many MCP sessions the daemon serves at the same time. One client is one "
            "session for as long as it stays connected, and a single application routinely "
            "opens several — a Claude Desktop chat and a Claude Code running inside it each "
            "spawn their own choto-bridge. The limit exists because nothing else bounds that "
            "number: a bridge left behind by a client that has gone away holds its session "
            "until the process is killed, so an unlimited daemon accumulates them. Reaching "
            "the limit is answered, not ignored — the newcomer is told so in a JSON-RPC error "
            "naming this setting, rather than having its connection closed in silence."
        ),
    )
    idle_exit_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "How long the daemon serves nobody before it exits, when something else holds "
            "its socket and can start it again. That is the launchd agent's mode: the socket "
            "belongs to launchd, stays reachable while the daemon is gone, and the first "
            "connection to it starts a fresh one — so the exit costs a start-up and buys back "
            "the memory the daemon's models occupy (a gigabyte of embedding vectors and "
            "compiled CoreML on an 8 GB machine, held around the clock for the minutes a day "
            "it is used). A daemon that bound the socket itself (`choto --socket` by hand) "
            "ignores this and serves forever: it *is* the endpoint, so leaving would take "
            "Choto off the machine rather than park it. Five minutes because the timer starts "
            "only once every client has disconnected — a conversation the user is reading "
            "holds its session open — and it spans the gap where a warm daemon still pays "
            "off: quitting Claude Desktop and opening it again, or a bridge restarted between "
            "conversations, finds the screen graph and the loaded models still there."
        ),
    )
    contention_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "How long a tool call waits for a resource another session is holding before it "
            "gives up and says so. Two things in the daemon admit exactly one holder: the "
            "screen (one pointer, one keyboard, one overlay) and the mission checklist "
            "(read-then-write decisions over one active mission). A caller that finds one of "
            "them taken waits — plans queue rather than interleave — but not without end: an "
            "MCP client abandons a request of its own accord after a minute or so, and a plan "
            "that started clicking after that would be moving the user's mouse with nobody "
            "left to read the result. Past this many seconds the call is refused by name "
            "instead, having touched nothing."
        ),
    )

    max_actions_per_plan: int = Field(
        default=MAX_ACTIONS_PER_PLAN,
        ge=1,
        le=MAX_ACTIONS_PER_PLAN,
        description=(
            "How many actions one run may perform: the plan's own steps plus the scrolls a "
            "target search spends looking below the fold. Lowering it makes runs more "
            "cautious in both, and a plan with more steps than this is refused before "
            "anything is clicked rather than run until the budget runs out mid-search. It "
            "cannot be raised past choto.config.MAX_ACTIONS_PER_PLAN, which is also the "
            "largest plan the models layer accepts."
        ),
    )

    overlay_enabled: bool = Field(
        default=True,
        description=(
            "Whether a run outlines the window it is working in with a translucent orange "
            "frame and a 'stop' button, drawn by a helper process (choto-overlay). The "
            "daemon acts on the user's own screen while the user is still sitting at it, so "
            "'which window is the machine in right now, and how do I stop it' must be "
            "answerable at a glance rather than by reading a log — the failsafe corner works "
            "only for someone who already knows it exists. The frame is excluded from screen "
            "capture and is click-through, so it changes nothing about what a plan sees or "
            "does; turning it off disables the helper process entirely, including the stop "
            "button (the failsafe corner is unaffected either way)."
        ),
    )

    icon_detection_enabled: bool = Field(
        default=True,
        description=(
            "Whether a freshly parsed window is also scanned for clickable glyphs, so a "
            "toolbar symbol with no text becomes an element the map can hold and an LLM can "
            "name once (annotate_icons). The scan is a YOLO11m head run through CoreML "
            "(choto.vision.icondetect): 43-50 ms per window and ~320 MB resident, flat in "
            "the window's size, against the several hundred milliseconds the OCR of the same "
            "window costs — and nothing at all on a cache hit, which is where repeat visits "
            "land. Turning it off stops new glyphs being learned; glyphs already labelled "
            "keep naming their elements, because those names live in the map, not in the "
            "detector. It is also off in effect on a machine with no detector installed "
            "(`choto model install --icon-detector PATH`), which is said once per retry "
            "window in the log rather than by failing a read."
        ),
    )
    icon_confidence: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum confidence a detection needs to become an icon candidate. Swept over "
            "four native screenshots (a Claude Desktop window, Docker Desktop, a Telegram "
            "chat list, a full-screen video); candidates surviving the size and text rules, "
            "summed: 84 at 0.05, 78 at 0.10, 74 at 0.15, 73 at 0.20, 71 at 0.25, 67 at 0.30, "
            "65 at 0.40, 64 at 0.50. Confidence does not sort real affordances from noise on "
            "this material — real ones appear at every score — so the default is the lowest "
            "swept value, OmniParser's own. Raising it to 0.25 costs the window's traffic "
            "lights (0.19), an overflow menu (0.05), a help button (0.22) and four radio "
            "bullets (0.07), while the noise it would remove is already removed by the two "
            "rules that are about shape rather than confidence. And it does not invent "
            "candidates where there is no interface: the video frame answers with nothing "
            "even at 0.05."
        ),
    )

    window_rect_refinement_enabled: bool = Field(
        default=True,
        description=(
            "Whether a located window's rectangle is sharpened against its title bar before "
            "the step works in it. The window server is authoritative about *which* window "
            "this is and only approximate about where it is — the rectangle it holds is the "
            "compositor's frame, not always the edges a user sees — and every consequence of "
            "the rectangle rides on it: which crop is hashed into a graph node, which "
            "elements count as the window's own, where a click is allowed to land. Measured "
            "on this machine: 15 ms per refinement on a 1710x1107 frame, 53 ms on a "
            "3420x2214 one, plus the ~25 ms capture that feeds it. It is paid once per "
            "window, not once per step: the outcome is kept against the "
            "application+title+rectangle+scale it was measured for, and a window whose "
            "server rectangle is unchanged has not moved or resized, so twenty steps in one "
            "window cost one measurement. Turning it off takes the surcharge off a slow "
            "display; the window server's own rectangle is then used as given, which is what "
            "every refusal to refine already falls back to (a window drawn without traffic "
            "lights — full screen, a tiling layout, GTK or Windows chrome — has no anchor to "
            "measure against and never will)."
        ),
    )

    motion_tracking_enabled: bool = Field(
        default=True,
        description=(
            "Whether a click re-reads its target's own pixels immediately before pressing, "
            "and follows the target while it moves. The rest of the executor assumes a screen "
            "that holds still once it has settled, and on anything animated — a list still "
            "gliding, a sheet sliding into place, a window being dragged — that assumption "
            "fails silently: the click lands on whatever slid into the coordinates the match "
            "was taken at. One region grab plus one template match (~15 ms, see "
            "click_lead_ms) is enough to say whether the target is still there, where it has "
            "moved to, and whether it is moving *now*; a target that is moving is then "
            "followed until it stops (ordinary click) or its motion is steady enough to aim "
            "ahead of (predictive click). Turning it off restores the behaviour that came "
            "before the feature exactly: no region grabs, no tracker, and a click at the "
            "coordinates the match reported."
        ),
    )
    click_lead_ms: float = Field(
        default=15.0,
        ge=0.0,
        description=(
            "How far ahead of a moving target a predictive click aims, in milliseconds: the "
            "time between the frame the target was last seen on and the click landing on the "
            "screen. Below the pixel grid for anything slower than ~0.03 px/ms, which is why "
            "a target moving slower than that is clicked where it stands. See the lead-time "
            "block in this module for what the default is made of."
        ),
    )

    phash_max_hamming: int = Field(
        default=6,
        ge=0,
        description="Max Hamming distance between pHashes to treat two screens as the same.",
    )
    expect_screen_change_hamming: int = Field(
        default=1,
        ge=0,
        description=(
            "Max Hamming distance between pHashes still considered 'the screen did not "
            "change' when verifying an expect.screen_changes condition. Deliberately much "
            "tighter than phash_max_hamming: node identity asks 'is this the same screen?', "
            "an expectation asks 'did anything at all happen?'. This is only the coarse "
            "half of that answer; see expect_screen_change_pixel_fraction."
        ),
    )
    expect_screen_change_pixel_fraction: float = Field(
        default=0.0002,
        ge=0.0,
        le=1.0,
        description=(
            "Min fraction of changed pixels (capture.frame_diff, on downscaled grayscale "
            "frames) that makes expect.screen_changes report 'the screen changed'. The "
            "fine-grained half of the signal: a pHash is a 64-bit summary of the whole "
            "screen and is structurally blind to local changes, so a new window opening "
            "over a blank document, a ticked checkbox or a typed word can leave it "
            "untouched while the pixels plainly moved. Doubles as the 'has anything "
            "happened yet' signal inside wait_until_stable, which asks the very same "
            "question of the frame a wait started from."
        ),
    )

    stable_frames_required: int = Field(
        default=3,
        ge=1,
        description=(
            "Consecutive low-diff polls required before a screen that has been seen "
            "moving is called 'settled'. Times stable_poll_interval_ms, this is the "
            "length of the quiet window that ends a wait; it must outlast a lull inside "
            "an animation (the longest measured span of visible motion is 86 ms)."
        ),
    )
    stable_diff_threshold: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description=(
            "Max normalized frame diff between two consecutive polls still considered "
            "'nothing is moving'. Deliberately looser than "
            "expect_screen_change_pixel_fraction: that one must not miss a real change, "
            "this one must not be held below the idle churn a live screen always has, or "
            "no step would ever settle before its timeout. See the stabilization block "
            "in this module."
        ),
    )
    stable_poll_interval_ms: int = Field(
        default=STABLE_POLL_INTERVAL_MS,
        ge=1,
        description=(
            "Minimum spacing between two stabilization polls, measured from the start of "
            "one look to the start of the next. One look (grab + diff) costs ~50 ms, so "
            "at the default the loop never sleeps and polls at the hardware floor; a "
            "larger value throttles it, a smaller one cannot speed it up."
        ),
    )
    settle_quiet_grace_ms: int = Field(
        default=1100,
        ge=0,
        description=(
            "How long a screen that has not changed at all since the wait began must stay "
            "that way before the wait returns. Separates 'the action had no visible "
            "effect' from 'the animation has not started yet' — a System Settings panel "
            "with image previews stays perfectly still for ~540 ms and as long as "
            "~1015 ms before it repaints. Capped by the step's own timeout. Paid by steps "
            "that have no other readiness signal; see settle_quiet_grace_watched_ms."
        ),
    )
    settle_in_flight_ceiling_ms: int = Field(
        default=2000,
        ge=0,
        description=(
            "How long the wait keeps going once it has reason to believe a transition is "
            "in flight — either the control under the click answered, or the screen is "
            "still moving. Both are the same question, so both share this ceiling. Without "
            "it the wait holds the step for its whole budget: a click whose only effect is "
            "its own highlight costs five seconds of nothing, and a panel with a running "
            "clock never stays still long enough to be called settled at all. Measured "
            "from the action, and still capped by the step's own timeout."
        ),
    )
    settle_quiet_grace_watched_ms: int = Field(
        default=STABLE_POLL_INTERVAL_MS,
        ge=0,
        description=(
            "The same window for a step whose own expectation is watching the screen "
            "afterwards. That check polls fresh pixels until the step's timeout, so it "
            "catches a late transition by itself and the blind wait in front of it is "
            "pure latency. One poll interval keeps a single look — enough to see a repaint "
            "already in flight — and hands everything later to the verification loop. Only "
            "applies when the pre-action screen does *not* already satisfy the "
            "expectation; see the stabilization block in this module."
        ),
    )

    map_unconfirmed_after_visits: int = Field(
        default=20,
        ge=1,
        description=(
            "How many further sightings of an application it takes before a window node "
            "seen once and never matched again is deleted. A node is a hypothesis — 'this "
            "picture is a window we will meet again' — and every later look at that "
            "application is a test of it: after this many tests without a single rematch, "
            "the hypothesis is refuted and the node is a fingerprint of a moment, not of a "
            "window. Counted in sightings rather than in elapsed time on purpose: a week "
            "in which the app was never opened proves nothing about the node, while twenty "
            "visits that all landed elsewhere prove a great deal. On the live map at the "
            "time of writing this removed 24 of 98 nodes, all of them windows whose "
            "contents had moved on (a save sheet with a different filename typed into it, "
            "a settings pane mid-animation)."
        ),
    )
    map_max_nodes_per_window: int = Field(
        default=3,
        ge=1,
        description=(
            "How many nodes one (application, window title) pair may keep; beyond that the "
            "oldest by last_seen_at are deleted. A node is identified by the perceptual "
            "hash of the window's crop, so a window whose contents change — a calculator "
            "display, a document being typed into, a save sheet — mints a new node every "
            "time it is read, and the map turns into a reel of snapshots of one window. "
            "Three keeps the same window recognizable across a couple of its states "
            "without letting one window own the map: on the live map, TextEdit's 'Save' "
            "sheet alone had grown to 14 nodes and the Calculator window to 8."
        ),
    )
    map_visits_per_app: int = Field(
        default=20_000,
        ge=1,
        description=(
            "How many sightings the visit journal keeps per application; past that the oldest "
            "are deleted, in the same pass that evicts nodes. Eviction deliberately leaves the "
            "journal alone — a node is a cache and a sighting is history — which makes it the "
            "one table in the graph with no ceiling of its own, and this is that ceiling. "
            "Sized from the live map: one day of heavy driving (33 runs over 20 hours) "
            "recorded 245 sightings across 6 applications, 119 of them for the busiest one, "
            "so a month at that intensity is ~3,600 rows for that application and this cap is "
            "over five such months — years of ordinary use, where a day holds a handful of "
            "runs rather than 33. At 234 bytes a row (measured on that same journal: the "
            "table plus its three indexes) it bounds the file too: ~5 MB per application, "
            "and only after months of continuous benchmark-grade driving. What a "
            "trim destroys is summarized before the rows go (how many were cut, and when the "
            "application was first seen at all), so a trimmed journal reports a smaller "
            "history rather than a false one. "
            "It may not be set below map_unconfirmed_after_visits, and the settings refuse to "
            "load if it is: the eviction rule counts an application's sightings *after* a "
            "node from this very journal, so a cap below that threshold could cut away the "
            "evidence the rule runs on and leave an unconfirmed node immortal. At or above "
            "it the rule is provably unaffected — rotation only ever removes rows older than "
            "every retained row, so a count taken from a moment inside the retained stretch "
            "is exact, and one taken from before it returns the whole retained stretch, which "
            "is the cap and therefore already clears the threshold."
        ),
    )

    fuzzy_match_threshold: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum rapidfuzz score (0-100) for a fuzzy target match.",
    )
    semantic_match_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a semantic (embedding) target match.",
    )

    platform_backend: str = Field(
        default=PLATFORM_AUTO,
        description=(
            "Which platform's implementations answer the OS seams (windows, input, "
            "applications, capture), by name from choto.platforms.PLATFORMS. The "
            "default 'auto' detects the platform from sys.platform; an explicit name "
            "pins it. An unknown or unported name is refused when the daemon builds "
            "its context, not on the first click."
        ),
    )

    ocr_engine: str = Field(
        default=OCR_ENGINE_APPLE_VISION,
        description=(
            "Which recognition engine reads the screen, by name from "
            "choto.ocr.engine.OCR_ENGINES. An unknown name is refused when the daemon "
            "builds its context, not on the first screen read. The remaining ocr_* "
            "settings below describe the two-pass recognition this project needs; an "
            "engine is free to reach it its own way, but must honour "
            "ocr_glyph_pass_enabled, which is how a caller asks for prose only."
        ),
    )
    ocr_languages: list[str] = Field(
        default_factory=lambda: ["en-US", "ru-RU"],
        description="BCP-47 language tags passed to Apple Vision OCR, in priority order.",
    )
    ocr_uses_language_correction: bool = Field(
        default=False,
        description=(
            "Whether Apple Vision applies language correction. Off by default: on the "
            "reference frame it changed no recall figure and cost 165 ms per parse."
        ),
    )
    ocr_text_revision: int = Field(
        default=3,
        ge=1,
        description=(
            "VNRecognizeTextRequest revision used for the primary whole-frame pass. Must be "
            "one of the revisions this macOS supports, or recognition raises OcrError."
        ),
    )
    ocr_glyph_pass_enabled: bool = Field(
        default=True,
        description=(
            "Whether to run the second, tiled pass that recovers isolated glyphs (keypads, "
            "toolbars, single-character labels). Turning it off halves parse latency and "
            "loses most single characters — see the OCR block in this module."
        ),
    )
    ocr_glyph_revision: int = Field(
        default=2,
        ge=1,
        description=(
            "VNRecognizeTextRequest revision used for the glyph pass. Older revisions keep "
            "single characters that the current one discards as non-words. Should macOS stop "
            "supporting it, the pass runs on the nearest available revision and says so once "
            "in the log: fewer glyphs read, rather than every parse failing."
        ),
    )
    ocr_glyph_tile_divisor: int = Field(
        default=3,
        ge=1,
        description=(
            "How small a glyph-pass crop is kept, as a fraction of the frame: no crop "
            "exceeds one divisor-th of the frame per axis. Raising it makes text larger "
            "relative to each crop (which is what makes small glyphs detectable) and "
            "multiplies OCR calls by its square when the whole frame has to be scanned."
        ),
    )
    ocr_glyph_magnify_factor: int = Field(
        default=3,
        ge=1,
        description=(
            "Magnification of the extra, single-crop glyph request taken per region of "
            "interest (1 disables it). Recovers marks whose absolute stroke is too thin to "
            "be seen at 1:1 — the Calculator minus sign is 4 px tall — at the cost of one "
            "more OCR request per region."
        ),
    )
    ocr_glyph_magnify_max_pixels: int = Field(
        default=1_000_000,
        ge=1,
        description=(
            "Pixel budget for one magnified glyph request: a region whose magnified area "
            "exceeds it is read at 1:1 only. Recognition cost grows with pixel count, so "
            "this is what keeps a full-screen window from paying for magnification."
        ),
    )
    ocr_glyph_tile_overlap_px: int = Field(
        default=48,
        ge=0,
        description=(
            "Pixels each glyph-pass tile extends past its grid cell, so no glyph is cut by "
            "a seam. Must exceed the tallest text expected to be read."
        ),
    )
    ocr_glyph_max_chars: int = Field(
        default=2,
        ge=1,
        description=(
            "Longest text kept from the glyph pass. Longer results are prose, which the "
            "primary pass reads in whole lines instead of tile-sized fragments."
        ),
    )
    ocr_duplicate_overlap_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the smaller bounding box that two boxes must share for the "
            "glyph-pass result to count as already-seen text and be dropped."
        ),
    )

    @model_validator(mode="after")
    def _journal_must_outlast_the_confirmation_window(self) -> Settings:
        if self.map_visits_per_app < self.map_unconfirmed_after_visits:
            raise ValueError(
                f"map_visits_per_app ({self.map_visits_per_app}) must be at least "
                f"map_unconfirmed_after_visits ({self.map_unconfirmed_after_visits}): the "
                "eviction rule counts an application's later sightings from the journal, and "
                "a cap below that threshold could trim away the evidence it needs, making an "
                "unconfirmed node immortal."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
