"""the visit journal outlives the nodes it points at

The map is about to stop growing forever: nodes that never matched a second time,
and the surplus snapshots of a window whose contents keep changing, are evicted
(``choto.graph.eviction``). Under the old schema that would also erase history —
``screen_visits.screen_id`` was ``NOT NULL`` with ``ON DELETE CASCADE``, so every
evicted node silently took its sightings with it. On the live database at the
time of writing, the first pass of the policy would have destroyed 78 of 245
recorded sightings.

"What was seen, and when" is not cache. It is the only record of where the
machine has actually been, ``recall`` dates its map from it, and the eviction
rule itself is defined in terms of it. So this revision makes the journal
independent of the graph:

* ``screen_id`` becomes nullable with ``ON DELETE SET NULL`` — the pointer to
  the node is released when the node goes, the row stays;
* ``app_name`` and ``window_title`` are copied onto every row, so a sighting
  whose node is gone can still say what was seen. They are backfilled from
  ``screens`` for the rows already stored.

The denormalization is load-bearing twice over. A released row would otherwise
be unreadable — "something was seen at 04:12" is not history. And the eviction
rule counts how many times an *application* was looked at after a given moment;
counted through a join, that number would shrink every time an earlier pass
evicted one of the nodes those looks landed on, so the policy would slowly stop
firing on exactly the applications it has already cleaned once.

SQLite cannot alter a column's nullability or a foreign key in place, so the
table is rebuilt: indexes dropped, the table renamed aside, the new one created,
rows copied through a ``LEFT JOIN`` onto ``screens``, the old one dropped and the
indexes recreated (plus one on ``app_name``, which the eviction query filters
by). Primary keys are carried over verbatim: ``id`` is what makes a sighting the
same sighting.

The join is a left one deliberately. A dangling ``screen_id`` cannot exist while
the foreign key is enforced, but SQLite only enforces it when the pragma is on,
and dropping a journal row here — in the very migration whose purpose is that
journal rows are never dropped — would be the wrong way to find out. Such a row
keeps its identity and gets an empty application name, which reads as "unknown"
everywhere the column is used.

Revision ID: a4e07f2b91c6
Revises: f3b8d206a17c
Create Date: 2026-07-27 06:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4e07f2b91c6"
down_revision: str | Sequence[str] | None = "f3b8d206a17c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "screen_visits"
_OLD_TABLE = "screen_visits_old"

# Written for a sighting whose node could not be found at all; see the module
# docstring. Also the default title of a window that has none.
_UNKNOWN = ""

_INDEXES = (
    ("ix_screen_visits_run_id", "run_id"),
    ("ix_screen_visits_screen_id", "screen_id"),
    ("ix_screen_visits_app_name", "app_name"),
)

_COLUMNS = (
    "id",
    "run_id",
    "screen_id",
    "app_name",
    "window_title",
    "seq",
    "step_index",
    "action",
    "seen_at",
    "created_at",
    "updated_at",
)


def _timestamps() -> list[sa.Column]:
    """The three timestamp columns every visit row carries."""
    return [
        sa.Column(
            "seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    """Rebuild ``screen_visits`` so a sighting survives its node's eviction."""
    for name, _column in _INDEXES[:2]:
        op.drop_index(name, table_name=_TABLE)
    op.rename_table(_TABLE, _OLD_TABLE)

    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("screen_id", sa.Integer(), nullable=True),
        sa.Column("app_name", sa.String(), nullable=False),
        sa.Column("window_title", sa.String(), nullable=False, server_default=_UNKNOWN),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.execute(
        sa.text(
            f"INSERT INTO {_TABLE} ({', '.join(_COLUMNS)}) "
            "SELECT old.id, old.run_id, old.screen_id, "
            "COALESCE(screens.app_name, :unknown), COALESCE(screens.window_title, :unknown), "
            "old.seq, old.step_index, old.action, old.seen_at, old.created_at, old.updated_at "
            f"FROM {_OLD_TABLE} AS old LEFT JOIN screens ON screens.id = old.screen_id"
        ).bindparams(unknown=_UNKNOWN)
    )

    op.drop_table(_OLD_TABLE)
    for name, column in _INDEXES:
        op.create_index(name, _TABLE, [column])


def downgrade() -> None:
    """Rebuild the old schema, dropping the sightings it cannot express.

    A row whose node has already been evicted has no ``screen_id`` to put back,
    and the old column is ``NOT NULL``: the pre-eviction schema simply has no
    place for a sighting that outlived its node. Those rows are therefore
    deleted, which is the one thing this revision exists to prevent — so
    downgrading past it is a decision to give up the history the eviction policy
    accumulated, not a neutral step back.
    """
    for name, _column in _INDEXES:
        op.drop_index(name, table_name=_TABLE)
    op.rename_table(_TABLE, _OLD_TABLE)

    op.create_table(
        _TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("screen_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        sa.text(
            f"INSERT INTO {_TABLE} "
            "(id, run_id, screen_id, seq, step_index, action, seen_at, created_at, updated_at) "
            "SELECT id, run_id, screen_id, seq, step_index, action, seen_at, created_at, updated_at "
            f"FROM {_OLD_TABLE} WHERE screen_id IS NOT NULL"
        )
    )
    op.drop_table(_OLD_TABLE)
    for name, column in _INDEXES[:2]:
        op.create_index(name, _TABLE, [column])
