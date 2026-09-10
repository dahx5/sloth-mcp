"""drop the nodes written when no window could be located

A read taken while the window server could not locate the app covers the whole
display: its elements belong to every application on screen at once, and it has
no window whose name they could be filed under. Such reads were nevertheless
stored as ordinary window nodes — titleless, holding the markup of the entire
desktop, and filed under whichever app happened to be frontmost. On a live
database that produced a "System Settings" node carrying a TextEdit save dialog
(``Format``, ``Where:``, ``Save``, ``Cancel``), and because it held more elements
than any real pane and had just been seen, ``recall`` opened the application's
map on it. The map lied in its most visible place.

The executor no longer writes such reads at all; this removes the ones already
stored.

**How they are identified.** ``screens.glyph_scope`` is the canonical signature
of the area OCR hunted single characters in, and the literal ``whole-frame`` is
written by exactly one situation: a parse handed no region of interest, which
happens only when no window rectangle was known. A scoped read always contributes
at least its own window's rectangle, so its signature is a list of ``x,y,w,h``
and can never take this form. The match is therefore causal rather than
heuristic, and false positives are not a matter of degree — no scoped read can
produce this value.

Two properties that look like alternatives are deliberately not used. The window
title is empty on these rows, but it is empty on honest untitled windows too, and
deleting by it would delete real memory. ``width``/``height`` are the dimensions
of the captured frame on *every* node — the coordinate space the elements live
in, not the window's size — so they separate nothing.

Rows written before ``d7a1c4f6b2e9`` cannot be caught by mistake: that migration
deleted every screen, so everything present was written under the window-node
rules.

Dependent rows are deleted explicitly rather than left to ``ON DELETE CASCADE``.
SQLite enforces foreign keys only while ``PRAGMA foreign_keys`` is on, and this
migration has to leave the same state either way. ``execution_runs`` is history,
not cache, and is untouched — only the visit rows pointing at the deleted nodes
go with them.

Revision ID: f3b8d206a17c
Revises: e5c93b71d4a2
Create Date: 2026-07-25 23:55:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f3b8d206a17c"
down_revision: str | Sequence[str] | None = "e5c93b71d4a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The glyph-scope signature of a parse that was pointed at no region of interest,
# i.e. one taken with no window located. Mirrors
# ``choto.executor.screen_reader.GLYPH_SCOPE_WHOLE_FRAME``; spelled out here
# because a migration is a snapshot of the schema as it was and must not change
# meaning when the application does.
_UNSCOPED_GLYPH_SCOPE = "whole-frame"

_DOOMED = "SELECT id FROM screens WHERE glyph_scope = :scope"


def upgrade() -> None:
    """Delete every node stored from a read that had no window, and its rows."""
    for statement in (
        f"DELETE FROM elements WHERE screen_id IN ({_DOOMED})",
        f"DELETE FROM screen_visits WHERE screen_id IN ({_DOOMED})",
        f"DELETE FROM edges WHERE from_screen IN ({_DOOMED}) OR to_screen IN ({_DOOMED})",
        "DELETE FROM screens WHERE glyph_scope = :scope",
    ):
        op.execute(sa.text(statement).bindparams(scope=_UNSCOPED_GLYPH_SCOPE))


def downgrade() -> None:
    """Restore nothing: the deleted rows were observations that cannot be undone.

    The graph is derived data — every honest node rebuilds itself on the next
    visit to its window, at the cost of one parse. The nodes removed here are the
    ones that were never derivable in the first place: whole-desktop readings
    filed under a single application. Re-creating them would mean re-creating the
    defect, and the pixels they came from are long gone in any case.

    The schema is unchanged by :func:`upgrade`, so there is nothing structural to
    reverse and downgrading past this revision is safe.
    """
