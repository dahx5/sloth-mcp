"""one node per (application, window hash), enforced by the database

The identity of a window node — "one row per (``app_name``, ``phash``)" — was
expressed by nothing but two non-unique indexes and the SELECT-then-INSERT
sequence in ``GraphRepository.upsert_screen``. That sequence is correct and,
today, unraced: the daemon is the only writer. Which is exactly the condition
under which an invariant stops being enforced by anything and nobody notices.

A second row for one pair is not a duplicate that wastes space, it is a *split
window*: ``find_screen_by_phash`` returns one candidate, so half of a window's
elements would be filed under a node no lookup ever reaches, and the map would
answer "this pane has three controls" about a pane with six.

Existing databases are collapsed before the constraint goes on, because a
migration that fails on a real database is a migration that cannot be run. Per
pair the freshest node (by ``last_seen_at``, id breaking ties) is kept; the
losers' edges and sightings are repointed onto it and the rows deleted, taking
their elements through ``ON DELETE CASCADE``. Their elements are deliberately
*not* merged into the keeper: two readings of one window are two element sets,
and interleaving them would produce a layout that was never on screen — the
keeper is the more recent reading and stands on its own. Edges that become
self-loops in the merge are dropped: an edge from a window to itself records no
transition.

The index on ``app_name`` goes with the change. The unique constraint's own index
leads with ``app_name``, so the old one is a strict prefix of it and answers
nothing it does not.

Revision ID: e2b95c17a4d3
Revises: c1e6a4d90b72
Create Date: 2026-07-28 12:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2b95c17a4d3"
down_revision: str | Sequence[str] | None = "c1e6a4d90b72"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCREENS = "screens"
_UNIQUE = "uq_screens_app_phash"
_APP_INDEX = "ix_screens_app_name"

# The survivor of each (application, hash) group: the most recently read row,
# with the id breaking a tie between two reads inside one clock tick.
_KEEPERS = """
CREATE TEMPORARY TABLE screen_merge AS
SELECT
    s.id AS loser,
    (
        SELECT k.id FROM screens k
        WHERE k.app_name = s.app_name AND k.phash = s.phash
        ORDER BY k.last_seen_at DESC, k.id DESC
        LIMIT 1
    ) AS keeper
FROM screens s
"""


def _collapse_duplicates() -> None:
    """Fold every rival node onto the freshest row of its (app, hash) pair."""
    op.execute(sa.text(_KEEPERS))
    for table, column in (
        ("edges", "from_screen"),
        ("edges", "to_screen"),
        ("screen_visits", "screen_id"),
    ):
        op.execute(
            sa.text(
                f"UPDATE {table} SET {column} = "
                f"(SELECT keeper FROM screen_merge WHERE loser = {table}.{column}) "
                f"WHERE {column} IN (SELECT loser FROM screen_merge WHERE loser <> keeper)"
            )
        )
    # A transition from a window to itself is not a transition; the merge is the
    # only thing that can produce one.
    op.execute(sa.text("DELETE FROM edges WHERE from_screen = to_screen"))
    op.execute(
        sa.text(
            "DELETE FROM screens WHERE id IN (SELECT loser FROM screen_merge WHERE loser <> keeper)"
        )
    )
    op.execute(sa.text("DROP TABLE screen_merge"))


def upgrade() -> None:
    """Collapse duplicate nodes, then make a duplicate impossible."""
    _collapse_duplicates()
    # Created as an index rather than through ``batch_alter_table``: adding a
    # table constraint in SQLite means recreating the table, and recreating
    # ``screens`` under ``PRAGMA foreign_keys=ON`` (``choto.graph.engine``) drops
    # it — taking every element with it and releasing every sighting's node. A
    # unique index is the same guarantee written where SQLite can add it in
    # place, and it is what the ORM's ``UniqueConstraint`` compiles to here.
    op.create_index(_UNIQUE, _SCREENS, ["app_name", "phash"], unique=True)
    op.drop_index(_APP_INDEX, table_name=_SCREENS)


def downgrade() -> None:
    """Give the plain application index back and let rivals exist again.

    Nothing is un-merged: the rows this collapsed were readings of one window
    that the map could only ever have served one of, and re-creating them would
    re-create the split.
    """
    op.create_index(_APP_INDEX, _SCREENS, ["app_name"])
    op.drop_index(_UNIQUE, table_name=_SCREENS)
