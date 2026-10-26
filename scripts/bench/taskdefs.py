from __future__ import annotations

from dataclasses import dataclass

from bench.taskops import (
    DARK_MODE_KEY,
    DOCK_AUTOHIDE_KEY,
    DOCK_DOMAIN,
    DOCK_TILESIZE_KEY,
    GLOBAL_DOMAIN,
    SANDBOX_DIR_NAME,
    CalculatorDisplayEquals,
    CheckOp,
    CloseSandboxTextEditDocuments,
    DefaultAtLeast,
    DefaultEquals,
    FileContains,
    FileContentEquals,
    MakeDir,
    PathIs,
    QuitApp,
    RemoveSandbox,
    RequireDefaultBelow,
    RequireDefaultIn,
    ResetSandbox,
    RestoreDefault,
    SetupOp,
    SnapshotDefault,
    SnapshotTextEditDocuments,
    TeardownOp,
    WindowTitleIsOneOf,
    WriteFile,
)

CALCULATOR = "Calculator"
TEXTEDIT = "TextEdit"
FINDER = "Finder"
SYSTEM_SETTINGS = "System Settings"

TIER_SIMPLE = "simple"
TIER_NAVIGATION = "navigation"
TIER_SETTINGS = "settings"
TIER_MULTI_APP = "multi-app"
TIER_WEAK_SPOT = "weak-spot"
TIER_LONG = "long"

TIERS = (
    TIER_SIMPLE,
    TIER_NAVIGATION,
    TIER_SETTINGS,
    TIER_MULTI_APP,
    TIER_WEAK_SPOT,
    TIER_LONG,
)

TAGS = frozenset({"text", "navigation", "scroll", "settings", "multi-app", "drag", "icons", "long"})

SANDBOX_PHRASE = f"папке {SANDBOX_DIR_NAME} в домашней папке"
SANDBOX_PHRASE_GEN = f"папки {SANDBOX_DIR_NAME} в домашней папке"

DOCK_TILESIZE_PASS = 110

DISPLAYS_TITLES = ("Displays", "Мониторы")
SOUND_TITLES = ("Sound", "Звук")
KEYBOARD_TITLES = ("Keyboard", "Клавиатура")
ABOUT_TITLES = ("About", "Об этом Mac")
GENERAL_TITLES = ("General", "Основные")


@dataclass(frozen=True)
class Task:
    id: str
    tier: str
    instruction: str
    apps: tuple[str, ...]
    tags: tuple[str, ...]
    rationale: str
    setup: tuple[SetupOp, ...]
    check: tuple[CheckOp, ...]
    teardown: tuple[TeardownOp, ...]

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"task {self.id!r}: unknown tier {self.tier!r}")
        unknown = set(self.tags) - TAGS
        if unknown:
            raise ValueError(f"task {self.id!r}: unknown tags {sorted(unknown)}")
        if not self.tags:
            raise ValueError(f"task {self.id!r}: at least one tag is required")
        if not self.check:
            raise ValueError(f"task {self.id!r}: a task without a check proves nothing")
        if not self.teardown:
            raise ValueError(f"task {self.id!r}: every task must clean up after itself")
        if not self.instruction.strip():
            raise ValueError(f"task {self.id!r}: instruction is empty")
        if not self.rationale.strip():
            raise ValueError(f"task {self.id!r}: rationale is empty")


def _long_document() -> str:
    filler = "routine filler text for the scrolling probe"
    lines = [f"line {index:03d}: {filler}" for index in range(1, 121)]
    lines[94] = "MARKER-LINE"
    return "\n".join(lines) + "\n"


LONG_DOCUMENT = _long_document()

STYLED_RTF = (
    "{\\rtf1\\ansi\\ansicpg1252\\cocoartf2761\n"
    "{\\fonttbl\\f0\\fswiss\\fcharset0 Helvetica;}\n"
    "\\pard\\ql\\f0\\fs28 Quarterly summary, first line.\\\n"
    "Quarterly summary, second line.}\n"
)


TASKS: tuple[Task, ...] = (
    Task(
        id="calc-multiply",
        tier=TIER_SIMPLE,
        instruction="Посчитай на калькуляторе 47 умножить на 89.",
        apps=(CALCULATOR,),
        tags=("text",),
        rationale=(
            "The shortest possible loop: a handful of clicks on labelled keys. Expected "
            "failure modes: the multiply and equals keys are glyphs that Vision reads "
            "inconsistently, and single-character targets such as '7' cannot be matched by "
            "substring at all (the matcher forbids it below three characters), so "
            "everything rides on exact OCR of one-character labels."
        ),
        setup=(QuitApp(CALCULATOR),),
        check=(CalculatorDisplayEquals("4183"),),
        teardown=(QuitApp(CALCULATOR),),
    ),
    Task(
        id="calc-percent",
        tier=TIER_SIMPLE,
        instruction="Посчитай на калькуляторе, сколько будет 15% от 240.",
        apps=(CALCULATOR,),
        tags=("text",),
        rationale=(
            "Same interface as calc-multiply, but the route is not written on the keys: the "
            "percent key means different things depending on the order it is pressed in. "
            "Fails when the sentence is translated into the wrong key sequence — a planning "
            "error rather than a vision error, which is why it is worth separating from the "
            "multiplication task."
        ),
        setup=(QuitApp(CALCULATOR),),
        check=(CalculatorDisplayEquals("36"),),
        teardown=(QuitApp(CALCULATOR),),
    ),
    Task(
        id="textedit-save",
        tier=TIER_SIMPLE,
        instruction=(
            "Создай в TextEdit новый документ с текстом «Отчёт готов» "
            f"и сохрани его в {SANDBOX_PHRASE} под именем note.txt."
        ),
        apps=(TEXTEDIT, FINDER),
        tags=("text", "navigation"),
        rationale=(
            "The check reads the file byte for byte, and that is the point: TextEdit starts "
            "in rich text, so a plan that types and saves without switching to plain text "
            "writes RTF markup under a .txt name — the run reports success and the check "
            "says otherwise, which is the false-success class this stand exists to catch. "
            "Also exercises the save sheet (choosing a destination folder is its own small "
            "navigation problem) and Cyrillic typing, which goes through the unicode input "
            "path rather than keycodes."
        ),
        setup=(ResetSandbox(), SnapshotTextEditDocuments()),
        check=(FileContentEquals("note.txt", "Отчёт готов", exact=True),),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox()),
    ),
    Task(
        id="finder-new-folder",
        tier=TIER_SIMPLE,
        instruction=f"Создай в {SANDBOX_PHRASE} новую папку с именем Reports.",
        apps=(FINDER,),
        tags=("text", "navigation"),
        rationale=(
            "Finder navigation to a named folder plus an inline rename field: a text box "
            "with no label and a very short life, since it closes on the first stray click. "
            "The usual failure leaves an 'untitled folder' behind because the rename never "
            "took, which the check names precisely — it looks for the directory by name, "
            "not for any new directory."
        ),
        setup=(ResetSandbox(),),
        check=(PathIs("Reports", "dir"),),
        teardown=(RemoveSandbox(),),
    ),
    Task(
        id="textedit-append",
        tier=TIER_SIMPLE,
        instruction=(
            f"Открой файл journal.txt из {SANDBOX_PHRASE_GEN}, допиши в конец "
            "новую строку «line 2: closed» и сохрани файл."
        ),
        apps=(TEXTEDIT, FINDER),
        tags=("text",),
        rationale=(
            "Editing an existing file rather than creating one: the plan has to open it, "
            "put the caret at the end (not at the beginning, where it lands) and save in "
            "place. Fails when the new line is inserted at the top, or when saving opens an "
            "unexpected sheet. Compared trimmed, because a trailing newline here is an "
            "editor habit and not a defect."
        ),
        setup=(
            ResetSandbox(),
            WriteFile("journal.txt", "line 1: opened\n"),
            SnapshotTextEditDocuments(),
        ),
        check=(FileContentEquals("journal.txt", "line 1: opened\nline 2: closed", exact=False),),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox()),
    ),
    Task(
        id="settings-displays",
        tier=TIER_NAVIGATION,
        instruction="Открой системные настройки и перейди в раздел «Мониторы».",
        apps=(SYSTEM_SETTINGS,),
        tags=("navigation",),
        rationale=(
            "The baseline navigation task: one sidebar item, visible without scrolling, "
            "verified from outside the app by the window title (System Settings renames its "
            "window after the pane it shows). Fails when the app reopens on the pane it was "
            "last left in and the plan assumed the sidebar's top."
        ),
        setup=(QuitApp(SYSTEM_SETTINGS),),
        check=(WindowTitleIsOneOf(SYSTEM_SETTINGS, DISPLAYS_TITLES),),
        teardown=(QuitApp(SYSTEM_SETTINGS),),
    ),
    Task(
        id="settings-sound-below-fold",
        tier=TIER_NAVIGATION,
        instruction="Открой системные настройки и перейди в раздел «Звук».",
        apps=(SYSTEM_SETTINGS,),
        tags=("navigation", "scroll"),
        rationale=(
            "The same shape as settings-displays, except the item sits below the fold of "
            "the sidebar. This is the exact case a live run escalated on once, which is why "
            "scroll_to_find exists; the task measures whether that recovery actually fires "
            "and lands on the right row, or overshoots and clicks a neighbour."
        ),
        setup=(QuitApp(SYSTEM_SETTINGS),),
        check=(WindowTitleIsOneOf(SYSTEM_SETTINGS, SOUND_TITLES),),
        teardown=(QuitApp(SYSTEM_SETTINGS),),
    ),
    Task(
        id="settings-keyboard",
        tier=TIER_NAVIGATION,
        instruction="Открой настройки клавиатуры.",
        apps=(SYSTEM_SETTINGS,),
        tags=("navigation", "text"),
        rationale=(
            "Deliberately says nothing about how to get there: the cheap route is the "
            "settings search field, the expensive one is scrolling the sidebar, and which "
            "one the model picks is part of what is being measured. The search route adds a "
            "failure mode of its own — the results list is a popover that steals the next "
            "click if the plan does not wait for it."
        ),
        setup=(QuitApp(SYSTEM_SETTINGS),),
        check=(WindowTitleIsOneOf(SYSTEM_SETTINGS, KEYBOARD_TITLES),),
        teardown=(QuitApp(SYSTEM_SETTINGS),),
    ),
    Task(
        id="settings-about",
        tier=TIER_NAVIGATION,
        instruction=(
            "Открой в системных настройках экран с информацией об этом Mac — "
            "где показаны модель, объём памяти и серийный номер."
        ),
        apps=(SYSTEM_SETTINGS,),
        tags=("navigation",),
        rationale=(
            "Two levels deep, and the instruction names the contents rather than the "
            "caption, so the model must map 'модель, память, серийный номер' onto a pane it "
            "either remembers or has to explore. Fails when the plan stops at the first "
            "level, which the title check catches: the window still reads 'General'."
        ),
        setup=(QuitApp(SYSTEM_SETTINGS),),
        check=(WindowTitleIsOneOf(SYSTEM_SETTINGS, ABOUT_TITLES),),
        teardown=(QuitApp(SYSTEM_SETTINGS),),
    ),
    Task(
        id="dark-mode-on",
        tier=TIER_SETTINGS,
        instruction="Включи тёмное оформление системы.",
        apps=(SYSTEM_SETTINGS,),
        tags=("settings", "navigation"),
        rationale=(
            "The first task whose result is a change to the machine rather than a file, and "
            "the first whose verification is a system command (defaults read -g "
            "AppleInterfaceStyle) with no screen involved at all. The appearance choice is "
            "a row of unlabelled preview images, so a plan that cannot name its target has "
            "to reach it some other way. Refuses to run when the machine is already dark: "
            "a task that could be passed by doing nothing measures nothing."
        ),
        setup=(
            SnapshotDefault(GLOBAL_DOMAIN, DARK_MODE_KEY),
            RequireDefaultIn(
                GLOBAL_DOMAIN,
                DARK_MODE_KEY,
                (None, "Light"),
                hint="Switch the system to light appearance and run the task again.",
            ),
            QuitApp(SYSTEM_SETTINGS),
        ),
        check=(DefaultEquals(GLOBAL_DOMAIN, DARK_MODE_KEY, "Dark"),),
        teardown=(RestoreDefault(GLOBAL_DOMAIN, DARK_MODE_KEY), QuitApp(SYSTEM_SETTINGS)),
    ),
    Task(
        id="dock-autohide-on",
        tier=TIER_SETTINGS,
        instruction="Сделай так, чтобы Док прятался автоматически.",
        apps=(SYSTEM_SETTINGS,),
        tags=("settings", "navigation"),
        rationale=(
            "A switch, not a value: the target is a toggle whose only text is the label "
            "beside it, so the plan has to click something the label merely points at. Also "
            "a live hazard for the executor — once autohide is on, the Dock disappears and "
            "the screen changes underneath any later step. Verified by defaults, so a "
            "toggle that looks flipped but did not commit still fails."
        ),
        setup=(
            SnapshotDefault(DOCK_DOMAIN, DOCK_AUTOHIDE_KEY),
            RequireDefaultIn(
                DOCK_DOMAIN,
                DOCK_AUTOHIDE_KEY,
                (None, "0"),
                hint="Turn Dock auto-hiding off and run the task again.",
            ),
            QuitApp(SYSTEM_SETTINGS),
        ),
        check=(DefaultEquals(DOCK_DOMAIN, DOCK_AUTOHIDE_KEY, "1"),),
        teardown=(RestoreDefault(DOCK_DOMAIN, DOCK_AUTOHIDE_KEY), QuitApp(SYSTEM_SETTINGS)),
    ),
    Task(
        id="dock-icon-size-max",
        tier=TIER_SETTINGS,
        instruction="Сделай значки в Доке максимально крупными.",
        apps=(SYSTEM_SETTINGS,),
        tags=("settings", "drag"),
        rationale=(
            "A slider — the one control that cannot be clicked into a value and has no text "
            "anywhere on it. The plan must express the gesture as a drag anchored to "
            "something nearby, which is precisely the offset-from-anchor path in the DSL. "
            "Fails when the drag is resolved to the wrong anchor, when it is released early, "
            "or when it moves the wrong slider on a pane that has several."
        ),
        setup=(
            SnapshotDefault(DOCK_DOMAIN, DOCK_TILESIZE_KEY),
            RequireDefaultBelow(
                DOCK_DOMAIN,
                DOCK_TILESIZE_KEY,
                DOCK_TILESIZE_PASS,
                hint="Make the Dock icons small again and run the task afresh.",
            ),
            QuitApp(SYSTEM_SETTINGS),
        ),
        check=(DefaultAtLeast(DOCK_DOMAIN, DOCK_TILESIZE_KEY, DOCK_TILESIZE_PASS),),
        teardown=(RestoreDefault(DOCK_DOMAIN, DOCK_TILESIZE_KEY), QuitApp(SYSTEM_SETTINGS)),
    ),
    Task(
        id="calc-to-file",
        tier=TIER_MULTI_APP,
        instruction=(
            "Посчитай на калькуляторе 128 умножить на 64 и сохрани получившееся число "
            f"в файл result.txt в {SANDBOX_PHRASE}."
        ),
        apps=(CALCULATOR, TEXTEDIT, FINDER),
        tags=("multi-app", "text"),
        rationale=(
            "The first task where a value has to survive a change of application. Every "
            "handover is a failure mode: the display is copied instead of retyped (or the "
            "other way round), focus moves before the copy lands, or the number arrives "
            "with the thousands separator the calculator drew. The check compares the file "
            "content, so a plausible-looking '8,192' fails."
        ),
        setup=(ResetSandbox(), QuitApp(CALCULATOR), SnapshotTextEditDocuments()),
        check=(FileContentEquals("result.txt", "8192", exact=False),),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox(), QuitApp(CALCULATOR)),
    ),
    Task(
        id="rename-from-doc",
        tier=TIER_MULTI_APP,
        instruction=(
            f"В {SANDBOX_PHRASE} есть файл target-name.txt, внутри которого записано имя. "
            "Переименуй файл unnamed-1.txt так, чтобы он назывался этим именем "
            "с расширением .txt."
        ),
        apps=(TEXTEDIT, FINDER),
        tags=("multi-app", "text"),
        rationale=(
            "The name is not in the instruction: it has to be read off the screen and then "
            "typed somewhere else, so the run depends on OCR being right about a word "
            "nobody can guess. Fails on a misread character (the check compares the exact "
            "file name), on the extension being dropped, and on the rename field closing "
            "before the text is committed."
        ),
        setup=(
            ResetSandbox(),
            WriteFile("target-name.txt", "quarterly-report\n"),
            WriteFile("unnamed-1.txt", "содержимое\n"),
            SnapshotTextEditDocuments(),
        ),
        check=(
            FileContentEquals("quarterly-report.txt", "содержимое", exact=False),
            PathIs("unnamed-1.txt", "absent"),
        ),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox()),
    ),
    Task(
        id="sum-two-files",
        tier=TIER_MULTI_APP,
        instruction=(
            f"В {SANDBOX_PHRASE} лежат файлы a.txt и b.txt, в каждом по числу. "
            "Сложи эти числа на калькуляторе и запиши сумму в новый файл sum.txt "
            "в той же папке."
        ),
        apps=(CALCULATOR, TEXTEDIT, FINDER),
        tags=("multi-app", "text", "long"),
        rationale=(
            "Two reads, one computation and one write, across three applications — the "
            "longest chain where every link is still individually simple. The interesting "
            "failure is arithmetic done in the model's head instead of on the calculator: "
            "the file would be right while the task it stands for was never performed. "
            "Nothing here can detect that, which is a limit of the check, stated openly."
        ),
        setup=(
            ResetSandbox(),
            WriteFile("a.txt", "1274\n"),
            WriteFile("b.txt", "3891\n"),
            QuitApp(CALCULATOR),
            SnapshotTextEditDocuments(),
        ),
        check=(FileContentEquals("sum.txt", "5165", exact=False),),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox(), QuitApp(CALCULATOR)),
    ),
    Task(
        id="rubber-band-move",
        tier=TIER_WEAK_SPOT,
        instruction=(
            f"В {SANDBOX_PHRASE} лежат три текстовых файла. "
            "Перемести их все в папку Archive внутри этой же папки."
        ),
        apps=(FINDER,),
        tags=("drag", "navigation"),
        rationale=(
            "A rubber-band selection starts on empty canvas — a place with no text on it — "
            "so it can only be expressed as an offset from an anchor, and the whole path is "
            "resolved before the button goes down. Fails when the anchor's offset lands on "
            "a file instead of beside it (dragging one file rather than framing three), or "
            "when the drop lands next to the folder rather than on it. The check names "
            "each file in both places, so a partial move reads as a partial move."
        ),
        setup=(
            ResetSandbox(),
            MakeDir("Archive"),
            WriteFile("notes-a.txt", "alpha\n"),
            WriteFile("notes-b.txt", "beta\n"),
            WriteFile("notes-c.txt", "gamma\n"),
        ),
        check=(
            PathIs("Archive/notes-a.txt", "file"),
            PathIs("Archive/notes-b.txt", "file"),
            PathIs("Archive/notes-c.txt", "file"),
            PathIs("notes-a.txt", "absent"),
            PathIs("notes-b.txt", "absent"),
            PathIs("notes-c.txt", "absent"),
        ),
        teardown=(RemoveSandbox(),),
    ),
    Task(
        id="settings-back-arrow",
        tier=TIER_WEAK_SPOT,
        instruction=(
            "Открой в системных настройках экран с информацией об этом Mac, "
            "а потом вернись на предыдущий экран."
        ),
        apps=(SYSTEM_SETTINGS,),
        tags=("icons", "navigation"),
        rationale=(
            "The way back is a bare chevron with no label anywhere near it, which is the "
            "documented hole in an OCR-first executor: v1 has no icon detector, so the "
            "expected outcome is an escalation naming the missing target. A pass would mean "
            "the model found another route (a keyboard shortcut, the sidebar) — worth "
            "knowing either way. The title check distinguishes 'went back' from 'closed the "
            "window', which look the same on screen."
        ),
        setup=(QuitApp(SYSTEM_SETTINGS),),
        check=(WindowTitleIsOneOf(SYSTEM_SETTINGS, GENERAL_TITLES),),
        teardown=(QuitApp(SYSTEM_SETTINGS),),
    ),
    Task(
        id="scroll-find-marker",
        tier=TIER_WEAK_SPOT,
        instruction=(
            f"Открой файл long.txt из {SANDBOX_PHRASE_GEN}, найди в нём строку MARKER-LINE, "
            "замени её целиком на DONE-LINE и сохрани файл."
        ),
        apps=(TEXTEDIT, FINDER),
        tags=("scroll", "text", "long"),
        rationale=(
            "The target sits at line 95 of 120 — off screen when the document opens, and "
            "reachable only by scrolling or by the app's own find. Two failure modes worth "
            "telling apart: the executor scrolls, re-reads and lands on it (the "
            "scroll_to_find path), or the plan uses Find and Replace and never looks at the "
            "text at all. The check reads both conditions — DONE-LINE present, MARKER-LINE "
            "gone — so replacing by appending does not pass."
        ),
        setup=(ResetSandbox(), WriteFile("long.txt", LONG_DOCUMENT), SnapshotTextEditDocuments()),
        check=(
            FileContains("long.txt", "DONE-LINE", present=True),
            FileContains("long.txt", "MARKER-LINE", present=False),
        ),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox()),
    ),
    Task(
        id="drag-onto-folder",
        tier=TIER_WEAK_SPOT,
        instruction=f"Перетащи файл invoice.txt в папку Inbox внутри {SANDBOX_PHRASE_GEN}.",
        apps=(FINDER,),
        tags=("drag",),
        rationale=(
            "A drop, as opposed to rubber-band-move's frame: both ends of the path are "
            "labelled, so resolving them is easy and the difficulty is entirely in the "
            "gesture. Fails when the drop lands between rows (the file goes nowhere and the "
            "run still reports success — a false success the check catches), or when the "
            "press-move-release is too fast for Finder to register a drag at all."
        ),
        setup=(ResetSandbox(), MakeDir("Inbox"), WriteFile("invoice.txt", "invoice 2026-07\n")),
        check=(PathIs("Inbox/invoice.txt", "file"), PathIs("invoice.txt", "absent")),
        teardown=(RemoveSandbox(),),
    ),
    Task(
        id="textedit-center-align",
        tier=TIER_WEAK_SPOT,
        instruction=(
            f"Открой файл styled.rtf из {SANDBOX_PHRASE_GEN}, выровняй весь его текст "
            "по центру и сохрани."
        ),
        apps=(TEXTEDIT, FINDER),
        tags=("icons", "text"),
        rationale=(
            "Alignment lives on a row of unlabelled toolbar buttons — four icons that "
            "differ only in the length of the lines drawn on them, which OCR cannot tell "
            "apart. There is a menu route with real words, so this measures whether the "
            "model reaches for text when pixels have nothing to say. Verified by reading "
            "the saved RTF for the centred-paragraph directive, which is invisible on "
            "screen and impossible to fake."
        ),
        setup=(ResetSandbox(), WriteFile("styled.rtf", STYLED_RTF), SnapshotTextEditDocuments()),
        check=(FileContains("styled.rtf", "\\qc", present=True),),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox()),
    ),
    Task(
        id="report-pipeline",
        tier=TIER_LONG,
        instruction=(
            f"В {SANDBOX_PHRASE} лежат файлы hours.txt и rate.txt, в каждом по числу. "
            "Создай там папку Payroll, посчитай на калькуляторе произведение этих чисел "
            "и сохрани в папке Payroll документ total.txt со строкой «total: <число>»."
        ),
        apps=(CALCULATOR, TEXTEDIT, FINDER),
        tags=("long", "multi-app", "text", "navigation"),
        rationale=(
            "Four stages that only count together: read two files, create a folder, compute, "
            "write into the folder just created. Fifteen-plus steps, and the plan's action "
            "budget (50) is a real ceiling once scroll recovery starts spending it. A "
            "mistake in the middle is unrecoverable — a document saved next to the folder "
            "rather than inside it fails the check while every individual step reports "
            "success. This is the task that says whether long autonomous batches are worth "
            "attempting at all."
        ),
        setup=(
            ResetSandbox(),
            WriteFile("hours.txt", "37\n"),
            WriteFile("rate.txt", "145\n"),
            QuitApp(CALCULATOR),
            SnapshotTextEditDocuments(),
        ),
        check=(
            PathIs("Payroll", "dir"),
            FileContentEquals("Payroll/total.txt", "total: 5365", exact=False),
        ),
        teardown=(CloseSandboxTextEditDocuments(), RemoveSandbox(), QuitApp(CALCULATOR)),
    ),
    Task(
        id="sandbox-triage",
        tier=TIER_LONG,
        instruction=(
            f"В {SANDBOX_PHRASE} лежат шесть файлов. Создай в ней две папки — Texts и "
            "Notes — и разложи файлы: с расширением .txt в Texts, с расширением .md в Notes."
        ),
        apps=(FINDER,),
        tags=("long", "drag", "navigation"),
        rationale=(
            "The same operation repeated until something drifts: two folders to create and "
            "six files to sort, in a window whose contents rearrange after every move. That "
            "is the point — a plan written against the layout seen at the start is wrong by "
            "the third file, so this measures whether targets are re-resolved against the "
            "screen as it is rather than as it was. Twelve independent checks, so a partial "
            "sort is reported as a partial sort rather than a flat failure."
        ),
        setup=(
            ResetSandbox(),
            WriteFile("alpha.txt", "alpha\n"),
            WriteFile("beta.txt", "beta\n"),
            WriteFile("gamma.txt", "gamma\n"),
            WriteFile("delta.md", "# delta\n"),
            WriteFile("epsilon.md", "# epsilon\n"),
            WriteFile("zeta.md", "# zeta\n"),
        ),
        check=(
            PathIs("Texts/alpha.txt", "file"),
            PathIs("Texts/beta.txt", "file"),
            PathIs("Texts/gamma.txt", "file"),
            PathIs("Notes/delta.md", "file"),
            PathIs("Notes/epsilon.md", "file"),
            PathIs("Notes/zeta.md", "file"),
            PathIs("alpha.txt", "absent"),
            PathIs("beta.txt", "absent"),
            PathIs("gamma.txt", "absent"),
            PathIs("delta.md", "absent"),
            PathIs("epsilon.md", "absent"),
            PathIs("zeta.md", "absent"),
        ),
        teardown=(RemoveSandbox(),),
    ),
)


def tasks_by_id() -> dict[str, Task]:
    catalog: dict[str, Task] = {}
    for task in TASKS:
        if task.id in catalog:
            raise ValueError(f"duplicate task id {task.id!r}")
        catalog[task.id] = task
    return catalog


def get_task(task_id: str) -> Task:
    catalog = tasks_by_id()
    task = catalog.get(task_id)
    if task is None:
        raise KeyError(f"unknown task {task_id!r}; known: {', '.join(sorted(catalog))}")
    return task
