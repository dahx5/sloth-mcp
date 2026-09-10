"""the map learns what is past the fold, without inventing where to click it

A window that scrolls holds more than one screenful, and the schema had no way
to say so. An element was a screen rectangle and nothing else, so a control a
flick away was either missing from the map or — the worse half — present with
the coordinates it happened to have at one scroll position. Those coordinates
are false at every other position: a plan clicking them lands on whatever has
since slid into the place.

So the row is split into what stays true and what does not.

* ``elements.doc_x`` / ``doc_y`` — where a control sits in the *document*, the
  frame anchored at the scrollable region's top-left as it stood at the first
  reading of the pass. Negative values are ordinary: they are what was above the
  fold when the pass began.
* ``elements.offscreen_side`` / ``offscreen_distance`` — which way a control
  lies from the viewport and how far the content must travel to bring it in.
  Set exactly when the control is not in view, and then ``x`` and ``y`` are
  ``NULL``. That is why ``x``/``y`` become nullable, and it is the whole point
  of the revision: the absence of a coordinate is storable, a wrong coordinate
  is not.
* ``elements.scroll_role`` — ``'document'`` for a control that travels with the
  content, ``'pinned'`` for furniture that held its screen position while the
  content moved under it (a sticky header, a toolbar), ``NULL`` for an element
  read with no scrolling knowledge at all.
* ``screens.scroll_covered_top`` / ``scroll_covered_bottom`` /
  ``scroll_viewport_h`` / ``scroll_ends`` — how much of the window's document has
  been in view and which edges the content was proved to run out at. All
  ``NULL`` (and ``[]``) means no scrolling pass ever covered this window, which
  is *not* the same statement as "this window does not scroll": the index has to
  keep "here there is nothing" apart from "nobody went there" (pro.md Part II
  §4a), and a node that silently read as complete would be the map passing its
  own ignorance off as knowledge.

Two check constraints hold the scheme up rather than a convention:
``ck_elements_offscreen_has_no_screen_box`` (a row has a screen position or a
direction to scroll, never both and never neither) and
``ck_elements_scroll_role_fields`` (each role keeps to its own columns). SQLite
cannot add a constraint or relax a ``NOT NULL`` in place, so ``elements`` is
rebuilt the way this project has rebuilt tables before: indexes dropped, table
renamed aside, the new one created, rows copied verbatim, the old one dropped,
indexes recreated. Primary keys are carried over — ``id`` is what makes an
element the same element, and the icon labelling pass addresses rows by it.

``screens`` is altered *in place*, one column at a time, and deliberately not
through ``batch_alter_table``. A batch that has to recreate a table drops it and
renames a copy in, and these migrations run over a connection with
``PRAGMA foreign_keys=ON`` (``choto.graph.engine``): dropping ``screens`` takes
every element and — through ``ON DELETE SET NULL`` — every sighting's node with
it. Adding and dropping a plain column needs no recreation, so nothing here goes
near that.

Nothing is backfilled. Every element already stored was read in one frame and is
on screen by construction, so it keeps its coordinates and gains no scrolling
claim: what the graph did not observe, this revision does not invent.

Revision ID: c1e6a4d90b72
Revises: b9f4e12c7a30
Create Date: 2026-07-28 09:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c1e6a4d90b72"
down_revision: str | Sequence[str] | None = "b9f4e12c7a30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "elements"
_OLD_TABLE = "elements_old"
_SCREENS = "screens"

_INDEXES = (
    ("ix_elements_screen_id", "screen_id"),
    ("ix_elements_norm_text", "norm_text"),
    ("ix_elements_icon_phash", "icon_phash"),
)

# The columns both shapes of the table share, and therefore the ones a rebuild
# copies. The new ones are absent from every existing row by definition.
_CARRIED = (
    "id",
    "screen_id",
    "text",
    "norm_text",
    "x",
    "y",
    "w",
    "h",
    "ocr_confidence",
    "embedding",
    "kind",
    "icon_phash",
    "created_at",
    "updated_at",
)

# No end has been proved for a window nobody scrolled — which is the honest
# reading of an empty set, not a claim that the content stops where it is shown.
_NO_ENDS = "[]"

_OFFSCREEN_CHECK = (
    "(offscreen_side IS NULL AND offscreen_distance IS NULL "
    " AND x IS NOT NULL AND y IS NOT NULL)"
    " OR (offscreen_side IS NOT NULL AND offscreen_distance IS NOT NULL "
    " AND x IS NULL AND y IS NULL AND scroll_role = 'document')"
)

_ROLE_CHECK = (
    "(scroll_role IS NULL AND doc_x IS NULL AND doc_y IS NULL AND offscreen_side IS NULL)"
    " OR (scroll_role = 'pinned' AND doc_x IS NULL AND doc_y IS NULL AND offscreen_side IS NULL)"
    " OR (scroll_role = 'document' AND doc_x IS NOT NULL AND doc_y IS NOT NULL)"
)


def _shared_columns(*, coordinates_required: bool) -> list[sa.Column]:
    """The columns every shape of ``elements`` has, in their stored order."""
    return [
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("screen_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("norm_text", sa.String(), nullable=False),
        sa.Column("x", sa.Integer(), nullable=not coordinates_required),
        sa.Column("y", sa.Integer(), nullable=not coordinates_required),
        sa.Column("w", sa.Integer(), nullable=False),
        sa.Column("h", sa.Integer(), nullable=False),
        sa.Column("ocr_confidence", sa.Float(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column(
            "kind",
            sa.Enum("text", "icon", name="elementkind", native_enum=False, length=8),
            nullable=False,
            server_default="text",
        ),
        sa.Column("icon_phash", sa.String(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _copy_carried(source: str, destination: str, where: str = "") -> None:
    """Move every row's shared columns from one shape of the table to the other."""
    columns = ", ".join(_CARRIED)
    op.execute(
        sa.text(f"INSERT INTO {destination} ({columns}) SELECT {columns} FROM {source} {where}")
    )


def upgrade() -> None:
    """Rebuild ``elements`` for scrolled knowledge and give nodes their extent."""
    for name, _column in _INDEXES:
        op.drop_index(name, table_name=_TABLE)
    op.rename_table(_TABLE, _OLD_TABLE)

    op.create_table(
        _TABLE,
        *_shared_columns(coordinates_required=False),
        sa.Column(
            "scroll_role",
            sa.Enum("document", "pinned", name="scrollrole", native_enum=False, length=8),
            nullable=True,
        ),
        sa.Column("doc_x", sa.Integer(), nullable=True),
        sa.Column("doc_y", sa.Integer(), nullable=True),
        sa.Column(
            "offscreen_side",
            sa.Enum("top", "bottom", "left", "right", name="scrolledge", native_enum=False,
                    length=6),
            nullable=True,
        ),
        sa.Column("offscreen_distance", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(_OFFSCREEN_CHECK, name="ck_elements_offscreen_has_no_screen_box"),
        sa.CheckConstraint(_ROLE_CHECK, name="ck_elements_scroll_role_fields"),
    )
    _copy_carried(_OLD_TABLE, _TABLE)
    op.drop_table(_OLD_TABLE)
    for name, column in _INDEXES:
        op.create_index(name, _TABLE, [column])

    op.add_column(_SCREENS, sa.Column("scroll_covered_top", sa.Integer(), nullable=True))
    op.add_column(_SCREENS, sa.Column("scroll_covered_bottom", sa.Integer(), nullable=True))
    op.add_column(_SCREENS, sa.Column("scroll_viewport_h", sa.Integer(), nullable=True))
    op.add_column(
        _SCREENS, sa.Column("scroll_ends", sa.JSON(), nullable=False, server_default=_NO_ENDS)
    )


def downgrade() -> None:
    """Restore the screen-only element and drop the rows it cannot express.

    An element that is past the fold has no screen coordinates to put back, and
    the old columns are ``NOT NULL``: the pre-scroll schema simply has no place
    for a control the map knows about and cannot point at. Those rows are
    therefore deleted, which is the one thing this revision exists to make
    possible — so going back is a decision to forget what is below the fold, not
    a neutral step. Everything in view, furniture included, keeps its row, its
    id and its coordinates; what it loses is the knowledge that it was ever part
    of a document longer than the window.
    """
    for name, _column in _INDEXES:
        op.drop_index(name, table_name=_TABLE)
    op.rename_table(_TABLE, _OLD_TABLE)

    op.create_table(
        _TABLE,
        *_shared_columns(coordinates_required=True),
        sa.ForeignKeyConstraint(["screen_id"], ["screens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    _copy_carried(_OLD_TABLE, _TABLE, where="WHERE x IS NOT NULL AND y IS NOT NULL")
    op.drop_table(_OLD_TABLE)
    for name, column in _INDEXES:
        op.create_index(name, _TABLE, [column])

    for column in (
        "scroll_ends",
        "scroll_viewport_h",
        "scroll_covered_bottom",
        "scroll_covered_top",
    ):
        op.drop_column(_SCREENS, column)
