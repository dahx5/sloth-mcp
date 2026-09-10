"""screens.glyph_scope: which areas the OCR glyph pass covered

The perceptual hash identifies a screen, but the stored element set also depends
on where the OCR glyph pass looked for lone characters. A parse made while
scanning one app's windows is not the same parse as one made over the whole
frame, so the area is recorded next to the elements and a reader asking for a
different area re-parses instead of reusing them.

Existing rows are backfilled with the empty string, which reads as "unknown
area" and matches no request: every screen already in the graph is re-parsed
once on its next visit and labelled properly from then on.

Revision ID: b41c7d0e92f5
Revises: cf90081ea3a0
Create Date: 2026-07-25 13:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b41c7d0e92f5"
down_revision: str | Sequence[str] | None = "cf90081ea3a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNKNOWN_GLYPH_SCOPE = ""


def upgrade() -> None:
    """Add ``screens.glyph_scope``, backfilled as unknown."""
    with op.batch_alter_table("screens", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "glyph_scope",
                sa.String(),
                nullable=False,
                server_default=_UNKNOWN_GLYPH_SCOPE,
            )
        )


def downgrade() -> None:
    """Drop ``screens.glyph_scope``, in place.

    Not through ``batch_alter_table``: a batch that has to recreate a table drops
    it and renames a copy in, and these migrations run over a connection with
    ``PRAGMA foreign_keys=ON`` (``choto.graph.engine``), where dropping
    ``screens`` cascades away every element and every edge. Dropping a plain
    column needs no recreation, and SQLite has done it in place since 3.35 — see
    the same note on ``d7a1c4f6b2e9``, which had the same defect.
    """
    op.drop_column("screens", "glyph_scope")
