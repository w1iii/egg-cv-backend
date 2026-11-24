"""Group completed inspections into saved batches.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("batches", sa.Column("id", sa.Integer, primary_key=True),
                    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
                    sa.Column("created_at", sa.DateTime, nullable=False))
    op.create_index("ix_batches_user_id", "batches", ["user_id"])
    op.create_table("batch_inspections",
                    sa.Column("batch_id", sa.Integer, sa.ForeignKey("batches.id", ondelete="CASCADE"), primary_key=True),
                    sa.Column("prediction_id", sa.Integer, sa.ForeignKey("predictions.id"), primary_key=True, unique=True))


def downgrade():
    op.drop_table("batch_inspections")
    op.drop_index("ix_batches_user_id", table_name="batches")
    op.drop_table("batches")
