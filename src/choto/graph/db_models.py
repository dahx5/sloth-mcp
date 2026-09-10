from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from choto.graph.map_models import ScrollEdge, ScrollRole
from choto.graph.missions import MissionItemStatus, MissionStatus
from choto.models import ElementKind, IconLabelSource, RunStatus


def _string_enum(enum: type[Enum], length: int) -> SAEnum:
    return SAEnum(
        enum,
        native_enum=False,
        length=length,
        values_callable=lambda members: [member.value for member in members],
    )


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
        list[Any]: JSON,
    }


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Screen(TimestampMixin, Base):
    __tablename__ = "screens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    app_name: Mapped[str] = mapped_column(String, nullable=False)
    phash: Mapped[str] = mapped_column(String, nullable=False)
    window_title: Mapped[str] = mapped_column(String, nullable=False, default="")
    anchors: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    is_stale: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    glyph_scope: Mapped[str] = mapped_column(String, nullable=False, default="")
    scroll_covered_top: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_covered_bottom: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_viewport_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_ends: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    elements: Mapped[list[Element]] = relationship(
        back_populates="screen",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_screens_phash", "phash"),
        Index("uq_screens_app_phash", "app_name", "phash", unique=True),
    )


class Element(TimestampMixin, Base):
    __tablename__ = "elements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    screen_id: Mapped[int] = mapped_column(
        ForeignKey("screens.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(String, nullable=False)
    norm_text: Mapped[str] = mapped_column(String, nullable=False)
    x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    w: Mapped[int] = mapped_column(Integer, nullable=False)
    h: Mapped[int] = mapped_column(Integer, nullable=False)
    ocr_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    kind: Mapped[ElementKind] = mapped_column(
        _string_enum(ElementKind, 8),
        nullable=False,
        default=ElementKind.TEXT,
        server_default=ElementKind.TEXT.value,
    )
    icon_phash: Mapped[str | None] = mapped_column(String, nullable=True)
    scroll_role: Mapped[ScrollRole | None] = mapped_column(
        _string_enum(ScrollRole, 8), nullable=True
    )
    doc_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doc_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    offscreen_side: Mapped[ScrollEdge | None] = mapped_column(
        _string_enum(ScrollEdge, 6), nullable=True
    )
    offscreen_distance: Mapped[int | None] = mapped_column(Integer, nullable=True)

    screen: Mapped[Screen] = relationship(back_populates="elements")

    __table_args__ = (
        Index("ix_elements_screen_id", "screen_id"),
        Index("ix_elements_norm_text", "norm_text"),
        Index("ix_elements_icon_phash", "icon_phash"),
        CheckConstraint(
            "(offscreen_side IS NULL AND offscreen_distance IS NULL "
            " AND x IS NOT NULL AND y IS NOT NULL)"
            " OR (offscreen_side IS NOT NULL AND offscreen_distance IS NOT NULL "
            " AND x IS NULL AND y IS NULL AND scroll_role = 'document')",
            name="ck_elements_offscreen_has_no_screen_box",
        ),
        CheckConstraint(
            "(scroll_role IS NULL AND doc_x IS NULL AND doc_y IS NULL "
            " AND offscreen_side IS NULL)"
            " OR (scroll_role = 'pinned' AND doc_x IS NULL AND doc_y IS NULL "
            " AND offscreen_side IS NULL)"
            " OR (scroll_role = 'document' AND doc_x IS NOT NULL AND doc_y IS NOT NULL)",
            name="ck_elements_scroll_role_fields",
        ),
    )


class IconGlyph(TimestampMixin, Base):
    __tablename__ = "icon_glyphs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    app_name: Mapped[str] = mapped_column(String, nullable=False)
    phash: Mapped[str] = mapped_column(String, nullable=False)
    crop_png: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False, default="", server_default="")
    label_source: Mapped[IconLabelSource] = mapped_column(
        _string_enum(IconLabelSource, 16),
        nullable=False,
        default=IconLabelSource.UNLABELED,
        server_default=IconLabelSource.UNLABELED.value,
    )

    __table_args__ = (UniqueConstraint("app_name", "phash", name="uq_icon_glyphs_app_phash"),)


class IconLabelScreen(Base):
    __tablename__ = "icon_label_screens"

    glyph_id: Mapped[int] = mapped_column(
        ForeignKey("icon_glyphs.id", ondelete="CASCADE"), primary_key=True
    )
    screen_id: Mapped[int] = mapped_column(
        ForeignKey("screens.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (Index("ix_icon_label_screens_screen_id", "screen_id"),)


class Edge(TimestampMixin, Base):
    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_screen: Mapped[int] = mapped_column(
        ForeignKey("screens.id", ondelete="CASCADE"), nullable=False
    )
    to_screen: Mapped[int] = mapped_column(
        ForeignKey("screens.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fail_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (Index("ix_edges_from_screen", "from_screen"),)


class ScreenVisit(TimestampMixin, Base):
    __tablename__ = "screen_visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("execution_runs.id", ondelete="SET NULL"), nullable=True
    )
    screen_id: Mapped[int | None] = mapped_column(
        ForeignKey("screens.id", ondelete="SET NULL"), nullable=True
    )
    app_name: Mapped[str] = mapped_column(String, nullable=False)
    window_title: Mapped[str] = mapped_column(String, nullable=False, default="")
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_screen_visits_run_id", "run_id"),
        Index("ix_screen_visits_screen_id", "screen_id"),
        Index("ix_screen_visits_app_name_seen_at", "app_name", "seen_at"),
    )


class AppJournal(TimestampMixin, Base):
    __tablename__ = "app_journals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    app_name: Mapped[str] = mapped_column(String, nullable=False)
    visits_cut: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (UniqueConstraint("app_name", name="uq_app_journals_app_name"),)


class ExecutionRun(TimestampMixin, Base):
    __tablename__ = "execution_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[RunStatus] = mapped_column(_string_enum(RunStatus, 16), nullable=False)
    journey: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Mission(TimestampMixin, Base):
    __tablename__ = "missions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[MissionStatus] = mapped_column(
        _string_enum(MissionStatus, 16),
        nullable=False,
        default=MissionStatus.ACTIVE,
        server_default=MissionStatus.ACTIVE.value,
    )

    items: Mapped[list[MissionItem]] = relationship(
        back_populates="mission",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MissionItem.seq",
    )

    __table_args__ = (
        Index(
            "uq_missions_one_active",
            "status",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
    )


class MissionItem(TimestampMixin, Base):
    __tablename__ = "mission_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_id: Mapped[int] = mapped_column(
        ForeignKey("missions.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(Text, nullable=False)
    acceptance: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[MissionItemStatus] = mapped_column(
        _string_enum(MissionItemStatus, 16),
        nullable=False,
        default=MissionItemStatus.PENDING,
        server_default=MissionItemStatus.PENDING.value,
    )
    fail_reason: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    recipe_json: Mapped[list[Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    fingerprint_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )

    mission: Mapped[Mission] = relationship(back_populates="items")

    __table_args__ = (UniqueConstraint("mission_id", "seq", name="uq_mission_items_seq"),)
