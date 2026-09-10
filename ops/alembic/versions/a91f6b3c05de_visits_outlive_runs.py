"""a sighting outlives its run, so runs can have a ceiling at all

``execution_runs`` was the one table in the store with no bound of any kind. A
row per plan the daemon executes, each carrying the whole plan and the whole
journey as JSON, and no rule anywhere — not eviction, not hygiene, not a single
``DELETE`` in ``src/`` — ever touching it. The docstring of ``graph/hygiene``
claimed ``screen_visits`` had been "the last table that could only grow"; it had
simply been the last one anybody looked at.

A ceiling could not be applied while ``screen_visits.run_id`` cascaded. Dropping
the oldest runs would have deleted the oldest sightings with them — trimming the
visit journal from its far end, by a rule that has nothing to do with the
journal, and destroying exactly what the revision ``a4e07f2b91c6`` exists to
preserve.

So ``run_id`` is released the same way ``screen_id`` was: nullable, ``ON DELETE
SET NULL``. A sighting keeps its application, its title, its order, its step and
its time; what a dropped run costs it is the ability to say which plan was being
executed. That is the same asymmetry the graph already draws — the map and the
plans are working material, "what was on screen and when" is history.

``screen_visits`` is rebuilt to change the constraint, since SQLite cannot alter
one in place. The rebuild is safe where a rebuild of ``screens`` would not be:
nothing references this table, so dropping the old shape under ``PRAGMA
foreign_keys=ON`` (``choto.graph.engine``) takes nothing with it. Rows are copied
verbatim, ids included — the id is what makes a sighting the same sighting.

Revision ID: a91f6b3c05de
Revises: f7d31a08e6b4
Create Date: 2026-07-28 12:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a91f6b3c05de"
down_revision: str | Sequence[str] | None = "f7d31a08e6b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "screen_visits"
_OLD_TABLE = "screen_visits_old"

_INDEXES = (
    ("ix_screen_visits_run_id", ["run_id"]),
    ("ix_screen_visits_screen_id", ["screen_id"]),
    ("ix_screen_visits_app_name_seen_at", ["app_name", "seen_at"]),
)

_CARRIED = (
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


def _columns(*, run_required: bool) -> list[sa.Column]:
    """The table's columns, differing only in whether a run must be named."""
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=not run_required),
        sa.Column("screen_id", sa.Integer(), nullable=True),
        sa.Column("app_name", sa.String(), nullable=False),
        sa.Column("window_title", sa.String(), nullable=False, server_default=""),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
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


def _rebuild(*, run_required: bool, run_ondelete: str, where: str = "") -> None:
    """Recreate the table with the given run-key rule and copy the rows across."""
    for name, _columns_of in _INDEXES:
        op.drop_index(name, table_name=_TABLE)
    op.rename_table(_TABLE, _OLD_TABLE)
    op.create_table(
        _TABLE,
        *_columns(run_required=run_required),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.id"], ondelete=run_ondelete),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    carried = ", ".join(_CARRIED)
    op.execute(
        sa.text(f"INSERT INTO {_TABLE} ({carried}) SELECT {carried} FROM {_OLD_TABLE} {where}")
    )
    op.drop_table(_OLD_TABLE)
    for name, columns in _INDEXES:
        op.create_index(name, _TABLE, columns)


def upgrade() -> None:
    """Let a sighting keep existing after its run is gone."""
    _rebuild(run_required=False, run_ondelete="SET NULL")


def downgrade() -> None:
    """Require a run again, and drop the sightings that no longer name one.

    The old shape has no room for a sighting whose run has aged out — the column
    is ``NOT NULL`` — so going back is a decision to forget exactly the rows this
    revision exists to keep. Every sighting that still names its run keeps its
    row and its id.
    """
    _rebuild(run_required=True, run_ondelete="CASCADE", where="WHERE run_id IS NOT NULL")
