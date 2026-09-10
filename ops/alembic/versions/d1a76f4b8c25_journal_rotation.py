"""the visit journal gains a ceiling, and a memory of what the ceiling cut

``screen_visits`` was the last table in the graph that could only grow. Nodes are
evicted (``choto.graph.eviction``) and the journal is deliberately spared, because
a node is a cache of what a window looks like and a sighting is the record that
the machine was there — but "never deleted" and "unbounded" are not the same
promise, and one run of the daemon writes a sighting per window it walks through.
On the live database a single day of driving added 245 of them.

The ceiling is per application and oldest-first
(``Settings.map_visits_per_app``). Trimming history is the operation that most
easily turns a report into a lie, so this revision adds, alongside it, the place
where what was trimmed survives as a summary: ``app_journals``, one row per
application, holding how many of its sightings were cut and when it was first
seen at all. Those are the two facts that stop being recoverable the moment the
rows go — everything else about the journal is still in the rows that remain.

Nothing is backfilled: no rotation has ever run, so no application has lost a
sighting, and a row asserting otherwise would be an invention. An application
with no row here has had nothing cut.

``ix_screen_visits_app_name`` is replaced by a composite over
``(app_name, seen_at)``. Both journal queries are "this application, in time
order" — how many sightings since a moment (the eviction rule) and which are the
oldest (rotation) — and the composite answers each without a sort while its
leading column still serves everything the old index did.

Revision ID: d1a76f4b8c25
Revises: c8f2a5d1e703
Create Date: 2026-07-27 14:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d1a76f4b8c25"
down_revision: str | Sequence[str] | None = "c8f2a5d1e703"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VISITS = "screen_visits"
_OLD_INDEX = "ix_screen_visits_app_name"
_NEW_INDEX = "ix_screen_visits_app_name_seen_at"


def upgrade() -> None:
    """Create the per-application journal summary and index the journal for it."""
    op.create_table(
        "app_journals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("app_name", sa.String(), nullable=False),
        sa.Column("visits_cut", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("app_name", name="uq_app_journals_app_name"),
    )
    op.drop_index(_OLD_INDEX, table_name=_VISITS)
    op.create_index(_NEW_INDEX, _VISITS, ["app_name", "seen_at"])


def downgrade() -> None:
    """Drop the summary and restore the single-column index.

    The summary is the only record that a trim ever happened; going back is a
    decision to forget that the journal was ever shorter than the history it
    describes. The sightings themselves are untouched here — they were removed,
    if they were, long before this step.
    """
    op.drop_index(_NEW_INDEX, table_name=_VISITS)
    op.create_index(_OLD_INDEX, _VISITS, ["app_name"])
    op.drop_table("app_journals")
