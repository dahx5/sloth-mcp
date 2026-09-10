"""Alembic environment.

The database URL and target metadata are sourced from the application itself
(:mod:`choto.config` and :mod:`choto.graph`) rather than hard-coded in
``alembic.ini``, so migrations always target the same store the app uses.
"""

from logging.config import fileConfig

from alembic import context

from choto.config import get_settings
from choto.graph.db_models import Base
from choto.graph.engine import create_engine_from_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI needed)."""
    engine = create_engine_from_settings(get_settings())
    context.configure(
        url=engine.url.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = create_engine_from_settings(get_settings())

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
