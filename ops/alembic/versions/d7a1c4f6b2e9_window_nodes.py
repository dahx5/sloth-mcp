"""graph nodes become windows: screens.window_title, cache rebuilt

The unit of memory changed from the whole display to a single application
window: a node's perceptual hash is now taken over that window's crop, its
elements are read from inside it, and its title is stored as a label so a plan
can name it (``target.window``) and a report can say which window a run worked
in.

Every existing row was keyed by a whole-screen hash and carries elements from
whatever else happened to be on screen, so no row can be reinterpreted under the
new rule — they are dropped rather than migrated. The graph is a cache of
derived observations: it rebuilds itself on the next visit to each window, at
the cost of one parse. ``execution_runs`` is history, not cache, and is kept.

The new index on ``app_name`` matches how nodes are now looked up: the closest
hash *within the same application*. Window crops are small and often nearly
uniform, so hashes that never collided across whole screens collide readily
across apps, and a cross-app hit would serve another program's elements.

Revision ID: d7a1c4f6b2e9
Revises: b41c7d0e92f5
Create Date: 2026-07-25 17:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d7a1c4f6b2e9"
down_revision: str | Sequence[str] | None = "b41c7d0e92f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NO_TITLE = ""


def upgrade() -> None:
    """Add ``screens.window_title``, index ``app_name``, and drop the stale cache."""
    with op.batch_alter_table("screens", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "window_title",
                sa.String(),
                nullable=False,
                server_default=_NO_TITLE,
            )
        )
    op.create_index("ix_screens_app_name", "screens", ["app_name"])

    # Order matters even with ON DELETE CASCADE: SQLite enforces foreign keys
    # only when the pragma is on, and this migration must leave the same state
    # either way.
    op.execute(sa.text("DELETE FROM edges"))
    op.execute(sa.text("DELETE FROM elements"))
    op.execute(sa.text("DELETE FROM screens"))


def downgrade() -> None:
    """Drop the window-node column and index.

    The nodes deleted by :func:`upgrade` are not restored — they were derived
    observations, and the pre-pivot readings that produced them no longer exist.
    Downgrading leaves an empty graph that the old code refills the old way.

    The column goes *in place*, and deliberately not through
    ``batch_alter_table``. A batch that has to recreate a table drops it and
    renames a copy in, and these migrations run over a connection with
    ``PRAGMA foreign_keys=ON`` (``choto.graph.engine``): dropping ``screens``
    takes every element and every edge with it through ``ON DELETE CASCADE``,
    and — once ``screen_visits`` exists — releases every sighting's node too.
    That is the trap ``c1e6a4d90b72`` documents and avoids for its own
    ``upgrade``, and it was live here: stepping a database with one node, one
    element and one edge through this function left the node and destroyed the
    other two. Dropping a plain column needs no recreation, and SQLite has done
    it in place since 3.35.
    """
    op.drop_index("ix_screens_app_name", table_name="screens")
    op.drop_column("screens", "window_title")
