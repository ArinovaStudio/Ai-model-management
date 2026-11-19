"""add caldav_event_uid and appointments table

Revision ID: d4b5c6e7f8a9
Revises: c3a4f891b72e
Create Date: 2026-10-26 13:20:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4b5c6e7f8a9'
down_revision: Union[str, Sequence[str], None] = 'c3a4f891b72e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add caldav_event_uid column to calendar_slots
    op.add_column(
        'calendar_slots',
        sa.Column('caldav_event_uid', sa.String(length=100), nullable=True)
    )
    op.create_index(
        'ix_calendar_slots_caldav_event_uid',
        'calendar_slots',
        ['caldav_event_uid'],
        unique=False
    )

    # 2. Create appointments table
    op.create_table(
        'appointments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('lead_id', sa.Integer(), nullable=True),
        sa.Column('slot_id', sa.Integer(), nullable=False),
        sa.Column('property_id', sa.Integer(), nullable=True),
        sa.Column('caldav_event_uid', sa.String(length=100), nullable=True),
        sa.Column('confirmation_status', sa.String(length=50), nullable=False, server_default='confirmed'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['lead_id'], ['leads.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['slot_id'], ['calendar_slots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['property_id'], ['property_listings.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_appointments_lead_id', 'appointments', ['lead_id'], unique=False)
    op.create_index('ix_appointments_slot_id', 'appointments', ['slot_id'], unique=False)
    op.create_index('ix_appointments_property_id', 'appointments', ['property_id'], unique=False)
    op.create_index('ix_appointments_caldav_event_uid', 'appointments', ['caldav_event_uid'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_appointments_caldav_event_uid', table_name='appointments')
    op.drop_index('ix_appointments_property_id', table_name='appointments')
    op.drop_index('ix_appointments_slot_id', table_name='appointments')
    op.drop_index('ix_appointments_lead_id', table_name='appointments')
    op.drop_table('appointments')

    op.drop_index('ix_calendar_slots_caldav_event_uid', table_name='calendar_slots')
    op.drop_column('calendar_slots', 'caldav_event_uid')
