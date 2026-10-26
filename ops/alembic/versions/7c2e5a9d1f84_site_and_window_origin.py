"""a node knows its site and where its window stood

Two columns and an identity that widens by one of them.

``screens.site`` is the host a browser window showed in its address bar (PR-037).
Pages of different sites are different nodes however alike they look, so the
site joins the unique identity: (application, site, hash). Existing rows get the
empty site — "not read" — and keep it until the next reading of that window
fills it in. Nothing here guesses a site from a title or an anchor: a guessed
site that is wrong would split one page into two nodes, or worse, keep two apart
that the next reading would have joined.

``screens.window_x`` / ``window_y`` are the window's top-left corner, in display
pixels, at the reading that wrote the node's elements (PR-046). From this
revision on an element is stored relative to that corner. Existing rows get
NULL: their elements were written in screen coordinates against a window
position nobody recorded, so they are left exactly as they are and a hit on such
a node is answered by a fresh reading, which rewrites them in the new frame.

``edges.success_count`` is zeroed (PR-042). Below this revision a step without
``expect`` counted as a confirmed transition, so the stored counts mix real
confirmations with unchecked arrivals and no column tells them apart. Zeroed
edges read as unverified and stay usable in routes (PR-021).

The unique index is swapped in place rather than through ``batch_alter_table``:
recreating ``screens`` under ``PRAGMA foreign_keys=ON`` would cascade away every
element and edge (see ``e2b95c17a4d3``).

Revision ID: 7c2e5a9d1f84
Revises: b3d92f7c41a8
Create Date: 2026-10-20 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7c2e5a9d1f84"
down_revision: str | Sequence[str] | None = "b3d92f7c41a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SCREENS = "screens"
_OLD_UNIQUE = "uq_screens_app_phash"
_NEW_UNIQUE = "uq_screens_app_site_phash"

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


def _collapse_sites() -> None:
    """Fold nodes that differ only by site onto the freshest one of each (app, hash)."""
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
    op.execute(sa.text("DELETE FROM edges WHERE from_screen = to_screen"))
    op.execute(
        sa.text(
            "DELETE FROM screens WHERE id IN (SELECT loser FROM screen_merge WHERE loser <> keeper)"
        )
    )
    op.execute(sa.text("DROP TABLE screen_merge"))


def upgrade() -> None:
    """Add the site and the window corner, both unknown on existing rows."""
    op.add_column(_SCREENS, sa.Column("site", sa.String(), nullable=False, server_default=""))
    op.add_column(_SCREENS, sa.Column("window_x", sa.Integer(), nullable=True))
    op.add_column(_SCREENS, sa.Column("window_y", sa.Integer(), nullable=True))
    op.create_index(_NEW_UNIQUE, _SCREENS, ["app_name", "site", "phash"], unique=True)
    op.drop_index(_OLD_UNIQUE, table_name=_SCREENS)
    # PR-042
    op.execute(sa.text("UPDATE edges SET success_count = 0"))


def downgrade() -> None:
    """Drop both columns; nodes that only a site kept apart become one again.

    Elements written relative to a window corner are moved back to screen
    coordinates by that corner, which is what the code below this revision reads.

    The edge confirmations upgrade zeroed are not restored: which of them were
    real was never recorded, so there is nothing to bring back.
    """
    op.execute(
        sa.text(
            "UPDATE elements SET "
            " x = x + (SELECT s.window_x FROM screens s WHERE s.id = elements.screen_id), "
            " y = y + (SELECT s.window_y FROM screens s WHERE s.id = elements.screen_id) "
            "WHERE x IS NOT NULL AND screen_id IN "
            " (SELECT id FROM screens WHERE window_x IS NOT NULL AND window_y IS NOT NULL)"
        )
    )
    _collapse_sites()
    op.create_index(_OLD_UNIQUE, _SCREENS, ["app_name", "phash"], unique=True)
    op.drop_index(_NEW_UNIQUE, table_name=_SCREENS)
    op.drop_column(_SCREENS, "window_y")
    op.drop_column(_SCREENS, "window_x")
    op.drop_column(_SCREENS, "site")
