from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from choto.config import Settings


def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _sqlite_url(db_path: Path) -> str:
    if str(db_path) == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    db_path = db_path.expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{db_path.resolve()}"


def create_engine_from_settings(settings: Settings) -> Engine:
    engine = create_engine(_sqlite_url(settings.db_path), future=True)
    event.listen(engine, "connect", _enable_sqlite_fk)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
