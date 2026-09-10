"""screen_visits: the ledger of which windows a run walked through

A run's ledger of seen windows existed only in memory: it was rendered into the
reply and then thrown away. Persisted, it turns into the graph's visit history —
how often each window is actually reached and when it was last reached — which
is what ``recall`` ranks its map by, and what lets "we were here 20 minutes ago"
outweigh "this node exists".

One row per (window, step) sighting, ordered by ``seq`` within the run, because
two reads can share a clock tick and timestamps alone would not preserve the
order the windows were met in.

Nothing is backfilled: the history starts now. Existing runs kept no such record,
and inventing visits for them would put fiction into the very data recall uses to
decide what to trust.

Revision ID: e5c93b71d4a2
Revises: d7a1c4f6b2e9
Create Date: 2026-07-25 20:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5c93b71d4a2"
down_revision: str | Sequence[str] | None = "d7a1c4f6b2e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``screen_visits`` with its run/screen lookup indexes."""
    op.create_table(
        "screen_visits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("screen_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["execution_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_screen_visits_run_id", "screen_visits", ["run_id"])
    op.create_index("ix_screen_visits_screen_id", "screen_visits", ["screen_id"])


def downgrade() -> None:
    """Drop ``screen_visits`` and its indexes."""
    op.drop_index("ix_screen_visits_screen_id", table_name="screen_visits")
    op.drop_index("ix_screen_visits_run_id", table_name="screen_visits")
    op.drop_table("screen_visits")
