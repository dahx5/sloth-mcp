"""missions: the graph learns to remember an intention, not only a screen

Every table before this one records what the *display* looked like. None of them
records what the model was trying to do, and that is the thing a long scenario
loses first: by the tenth window of a QA pass, the plan and — far worse — the
reasons behind each step have scrolled out of the context that wrote them, and
the run finishes against a confident recollection of a scenario nobody is still
holding.

``missions`` is the goal; ``mission_items`` is the checklist. Each item carries
three texts, and the middle one is why the table exists: ``title`` (what to do),
``intent`` (why this is in the list at all), ``acceptance`` (how the executor
knows it worked). All three are ``NOT NULL``, because an item missing any of
them reads, an hour later, as an instruction with no context — which is the
state this revision is meant to end.

**At most one mission is active, and the database is what says so.** The partial
unique index over ``status = 'active'`` is the whole invariant: "which checklist
am I on" must have exactly one answer, and a check in application code is a check
that two writers can walk through together. The repository checks as well, but
only to raise something a model can act on ("finish the current one"); this index
is what makes the rule true.

``recipe_json`` is the pass that actually worked and ``fingerprint_json`` is the
world it worked in — both JSON, both nullable, and both opaque to the schema.
NULL means "not recorded", which is deliberately distinguishable from a stored
JSON ``null``.

Nothing is backfilled: no mission has ever existed, and the tables start empty.

Revision ID: b9f4e12c7a30
Revises: d1a76f4b8c25
Create Date: 2026-07-27 16:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b9f4e12c7a30"
down_revision: str | Sequence[str] | None = "d1a76f4b8c25"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MISSIONS = "missions"
_ITEMS = "mission_items"

# The one-active invariant, as a partial unique index. The predicate is a
# literal token rather than a parameter because this is DDL, and the token is
# the enum value the ORM stores (``MissionStatus.ACTIVE``).
_ONE_ACTIVE = "uq_missions_one_active"
_ACTIVE = "active"

_MISSION_STATUS = sa.Enum(
    "active", "done", "abandoned", name="missionstatus", native_enum=False, length=16
)
_ITEM_STATUS = sa.Enum(
    "pending",
    "passed",
    "failed",
    "skipped",
    name="missionitemstatus",
    native_enum=False,
    length=16,
)


def upgrade() -> None:
    """Create the mission tables and the index that keeps one mission active."""
    op.create_table(
        _MISSIONS,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("status", _MISSION_STATUS, nullable=False, server_default=_ACTIVE),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        _ONE_ACTIVE,
        _MISSIONS,
        ["status"],
        unique=True,
        sqlite_where=sa.text(f"status = '{_ACTIVE}'"),
    )
    op.create_table(
        _ITEMS,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("mission_id", sa.Integer(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("acceptance", sa.Text(), nullable=False),
        sa.Column("status", _ITEM_STATUS, nullable=False, server_default="pending"),
        sa.Column("fail_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("recipe_json", sa.JSON(), nullable=True),
        sa.Column("fingerprint_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["mission_id"], [f"{_MISSIONS}.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Identity of a position within a checklist, and the lookup path for
        # "this mission's items in order" — the only query the table has.
        sa.UniqueConstraint("mission_id", "seq", name="uq_mission_items_seq"),
    )


def downgrade() -> None:
    """Drop both tables, and with them every intention ever recorded.

    Unlike the map, this is not a cache with a slower way to rebuild it: a
    mission is what a model meant to do, and once the tables are gone there is
    no observation that reconstructs it. The step exists to keep the revision
    chain walkable in both directions, and it says plainly what walking back
    costs.

    Items go first so the drop is valid under enforced foreign keys, whatever
    order the database would have chosen on its own.
    """
    op.drop_table(_ITEMS)
    op.drop_index(_ONE_ACTIVE, table_name=_MISSIONS)
    op.drop_table(_MISSIONS)
