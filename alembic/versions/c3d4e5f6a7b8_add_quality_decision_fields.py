"""Add per-egg quality decision and per-inspection decision counts.

Additive and non-destructive: no column is dropped or renamed and no
historical value is rewritten. Rows written before this revision keep
``decision`` NULL ("not recorded") rather than receiving a back-filled
decision under a policy that did not exist when they were inspected.

Revision ID: c3d4e5f6a7b8
Revises: batch_recording
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "batch_recording"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "detection_boxes", sa.Column("decision", sa.String(20), nullable=True)
    )
    op.add_column(
        "detection_boxes",
        sa.Column("decision_reason", sa.String(255), nullable=True),
    )
    op.create_index(
        "ix_detection_boxes_decision", "detection_boxes", ["decision"], unique=False
    )
    op.create_check_constraint(
        "ck_detection_boxes_decision",
        "detection_boxes",
        "decision IS NULL OR "
        "decision IN ('Acceptable', 'Reject', 'Needs Review')",
    )
    for name in ("acceptable_count", "reject_count"):
        op.add_column(
            "predictions",
            sa.Column(name, sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    for name in ("reject_count", "acceptable_count"):
        op.drop_column("predictions", name)
    op.drop_constraint(
        "ck_detection_boxes_decision", "detection_boxes", type_="check"
    )
    op.drop_index("ix_detection_boxes_decision", table_name="detection_boxes")
    op.drop_column("detection_boxes", "decision_reason")
    op.drop_column("detection_boxes", "decision")
