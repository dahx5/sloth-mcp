"""elements.embedding: a cache nothing ever wrote to and nothing ever read

The column was reserved for a per-element vector so the semantic stage of target
matching would not have to recompute one. It never happened: the matcher embeds
its candidates on the fly (``model2vec``, milliseconds on CPU), and no code path
in the service has ever assigned this column or selected it. What it holds on
every real database is ``NULL``, once per element row.

A field that is neither written nor read is not a slot held open for later, it is
a claim in the schema that the map keeps something it does not. Removing it costs
nothing that can be lost — there is no value in any row to lose — and it is put
back verbatim by the downgrade for the same reason.

SQLite has dropped columns in place since 3.35, and this one is in no index and
in neither check constraint, so nothing here goes near the table rebuild that
``PRAGMA foreign_keys=ON`` makes dangerous.

Revision ID: f7d31a08e6b4
Revises: e2b95c17a4d3
Create Date: 2026-07-28 12:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f7d31a08e6b4"
down_revision: str | Sequence[str] | None = "e2b95c17a4d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ELEMENTS = "elements"
_COLUMN = "embedding"


def upgrade() -> None:
    """Drop the unused embedding cache."""
    op.drop_column(_ELEMENTS, _COLUMN)


def downgrade() -> None:
    """Put the column back, empty — which is the only state it was ever in."""
    op.add_column(_ELEMENTS, sa.Column(_COLUMN, sa.LargeBinary(), nullable=True))
