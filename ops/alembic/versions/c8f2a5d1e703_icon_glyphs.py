"""icons: elements gain a kind, and a per-app vocabulary of glyphs appears

Choto reads screens with OCR, so a toolbar glyph with no caption was simply not
in the map: a plan could not name it and the executor could not click it. The
detector now finds where such ink sits, and this revision is where the finding
is kept.

Two changes, one idea. ``elements`` gains ``kind`` — ``'text'`` for everything
OCR read, ``'icon'`` for a detected glyph — and ``icon_phash``, the hash of the
drawing that was found there. An icon element is born with an empty ``text``
and is named later, which is exactly why the kind has to be recorded: a text
element with no text is debris to be ignored, while an icon element with no text
is a known place awaiting its name. Every row that already exists is backfilled
as ``'text'``, which is what it is.

``icon_glyphs`` is the vocabulary: one row per *distinct* drawing per
application, holding the crop and, once it has one, the label. Distinct rather
than per position because deciding what a picture means is the expensive step
and must be paid once — a toolbar repeating the same chevron eight times is
eight click targets and one question. Keyed per application because identical
pixels mean different things in different programs (a circled arrow is "reload"
in a browser and "retry" in a mail client), and a label leaking across that
boundary would put a confident wrong word on a control; the price is labelling a
generic glyph once per app, and it is accepted.

``crop_png`` is stored rather than re-cut on demand: the frame it came from is
gone by the time anyone asks, and a 20 px glyph has no resolution to spare for a
re-render.

Nothing is backfilled into ``icon_glyphs`` — no detection has ever run — and no
element is retro-classified as an icon. Both are the same rule: what the graph
did not observe, this revision does not invent.

Revision ID: c8f2a5d1e703
Revises: a4e07f2b91c6
Create Date: 2026-07-27 11:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8f2a5d1e703"
down_revision: str | Sequence[str] | None = "a4e07f2b91c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# What every element written before icons existed is, and the default for
# anything written by a caller that does not say.
_TEXT_KIND = "text"

# An unlabelled glyph and an unknown source are both the empty string: the two
# columns always move together, so one absent value would be a contradiction.
_UNLABELED = ""

_GLYPH_INDEX = "ix_elements_icon_phash"


def upgrade() -> None:
    """Classify existing elements as text and create the glyph vocabulary."""
    with op.batch_alter_table("elements", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "kind",
                sa.Enum("text", "icon", name="elementkind", native_enum=False, length=8),
                nullable=False,
                server_default=_TEXT_KIND,
            )
        )
        batch_op.add_column(sa.Column("icon_phash", sa.String(), nullable=True))
    op.create_index(_GLYPH_INDEX, "elements", ["icon_phash"])

    op.create_table(
        "icon_glyphs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("app_name", sa.String(), nullable=False),
        sa.Column("phash", sa.String(), nullable=False),
        sa.Column("crop_png", sa.LargeBinary(), nullable=False),
        sa.Column("label", sa.String(), nullable=False, server_default=_UNLABELED),
        sa.Column(
            "label_source",
            sa.Enum("", "llm", "tooltip", name="iconlabelsource", native_enum=False, length=16),
            nullable=False,
            server_default=_UNLABELED,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("app_name", "phash", name="uq_icon_glyphs_app_phash"),
    )


def downgrade() -> None:
    """Drop the vocabulary and the two element columns.

    The labels go with it, and they are the expensive part — each was a round
    trip to an LLM. Downgrading past this revision is therefore a decision to
    re-learn every icon, not a neutral step back. The element rows themselves
    survive: an icon element loses only the two columns that made it an icon,
    and OCR lines are untouched.
    """
    op.drop_table("icon_glyphs")
    op.drop_index(_GLYPH_INDEX, table_name="elements")
    with op.batch_alter_table("elements", schema=None) as batch_op:
        batch_op.drop_column("icon_phash")
        batch_op.drop_column("kind")
