"""a glyph's name records the screens it was given about

An icon glyph is keyed per application, and that is right: identical pixels mean
different things in different programs. What the key does not separate — and must
not, because separating it is unaffordable — is the same drawing meaning
different things inside *one* program. A game's paging arrow was named "next
character" on the screen that picks a character, and the map put those words on
the very same arrow on the screen that picks a dream, where nothing is a
character. A plan reading that is aiming at a control by a name whose meaning is
not on the screen, and nothing in the reply said so.

Narrowing the key to the window is not the fix. A window node is a pHash bucket,
so that game mints one per character and the name would have to be bought again
for each; and a node is an evictable cache entry while a label is knowledge that
cost a round trip to an LLM, so the name would die with the cache.

So the name stays the application's and gains a *provenance*: ``icon_label_screens``
lists the window nodes that were drawing a glyph when it was named — the screens
the labelling sheet showed, which are the screens the answer was given in sight
of. A position on any other node still gets the name, because a nameless icon is
unclickable and re-asking per window is what this scheme exists to avoid, but it
gets it marked as carried over (``choto.graph.icons.icon_element_text``). The
marker rides in the element's text because the text is the whole of what a window
reading shows for an icon, and it is 16 characters because the matcher resolves a
target inside a line that adds at most 20 (``MAX_CONTAINMENT_EXTRA_CHARS``): the
name a model is handed must go on resolving, or the hedge would cost more than
the lie.

Both foreign keys cascade. The ``screens`` end is why this is a table rather than
a list in a column: a node is evicted routinely, and a stored id outliving the
node it names would be a claim about a screen nobody holds — worse, SQLite hands
a deleted row's id to the next one, so it could later match an unrelated window
and re-assert exactly what this exists to flag.

The backfill reconstructs the home of every name already written, from the
positions that name was propagated onto: those are the places that were on the
map when it was applied, which is the same set ``apply_glyph_labels`` records
today. It is not an invention — it is the state the label was given about — and
it is what keeps an already-annotated application reading exactly as it did
yesterday instead of coming back hedged everywhere.

Nothing else is touched. In particular no element text is rewritten on the way
up: a name whose home is reconstructed is at home where it is drawn, and there is
nothing to mark.

Revision ID: b3d92f7c41a8
Revises: a91f6b3c05de
Create Date: 2026-08-02 12:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3d92f7c41a8"
down_revision: str | Sequence[str] | None = "a91f6b3c05de"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "icon_label_screens"
_INDEX = "ix_icon_label_screens_screen_id"

# What a carried-over name appends to itself on a window it was not given about;
# the constant lives in ``choto.graph.icons`` and is repeated here because a
# migration must keep saying what it said on the day it ran.
_BORROWED = " (borrowed name)"

# The home of a name already written: every window of the same application that
# draws the glyph in view. Out-of-view positions are excluded exactly as the
# labelling sheet excludes them — a copy below the fold was on no legend line, so
# the answer was not given in sight of it.
_BACKFILL = sa.text(
    "INSERT INTO icon_label_screens (glyph_id, screen_id) "
    "SELECT DISTINCT g.id, e.screen_id "
    "FROM icon_glyphs g "
    "JOIN elements e ON e.icon_phash = g.phash "
    " AND e.kind = 'icon' AND e.offscreen_side IS NULL "
    "JOIN screens s ON s.id = e.screen_id AND s.app_name = g.app_name "
    "WHERE g.label != ''"
)

# On the way back down the marker means nothing to the code that reads it, so it
# would be a permanent decoration on a control's name — and one long enough to
# matter to the matcher. Stripped from exactly the rows this scheme could have
# written it onto.
_STRIP_MARKER = sa.text(
    "UPDATE elements SET "
    " text = substr(text, 1, length(text) - :width), "
    " norm_text = substr(norm_text, 1, length(norm_text) - :width) "
    "WHERE kind = 'icon' AND text LIKE :pattern"
).bindparams(width=len(_BORROWED), pattern=f"%{_BORROWED}")


def upgrade() -> None:
    """Create the provenance table and reconstruct it for the names already given."""
    op.create_table(
        _TABLE,
        sa.Column("glyph_id", sa.Integer(), nullable=False),
        sa.Column("screen_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["glyph_id"], ["icon_glyphs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("glyph_id", "screen_id"),
    )
    op.create_index(_INDEX, _TABLE, ["screen_id"])
    op.get_bind().execute(_BACKFILL)


def downgrade() -> None:
    """Drop the provenance and take its marker off the names that carry it.

    The names themselves survive — they are the expensive part — and every one of
    them goes back to being asserted on every window of its application, which is
    what the code below this revision does with them.
    """
    op.get_bind().execute(_STRIP_MARKER)
    op.drop_index(_INDEX, table_name=_TABLE)
    op.drop_table(_TABLE)
