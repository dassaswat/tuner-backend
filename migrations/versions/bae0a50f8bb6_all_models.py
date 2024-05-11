"""all-models

Revision ID: bae0a50f8bb6
Revises: 
Create Date: 2024-05-10 10:21:00.600343

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bae0a50f8bb6'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('playlist_datasets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('playlist_id', sa.String(), nullable=False),
    sa.Column('energy', sa.Integer(), nullable=False),
    sa.Column('liveness', sa.Integer(), nullable=False),
    sa.Column('tempo', sa.Integer(), nullable=False),
    sa.Column('speechiness', sa.Integer(), nullable=False),
    sa.Column('acousticness', sa.Integer(), nullable=False),
    sa.Column('instrumentalness', sa.Integer(), nullable=False),
    sa.Column('danceability', sa.Integer(), nullable=False),
    sa.Column('loudness', sa.Integer(), nullable=False),
    sa.Column('valence', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_playlist_datasets_id'), 'playlist_datasets', ['id'], unique=False)
    op.create_index(op.f('ix_playlist_datasets_playlist_id'), 'playlist_datasets', ['playlist_id'], unique=True)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('spotify_id', sa.String(), nullable=False),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_spotify_id'), 'users', ['spotify_id'], unique=True)
    op.create_table('tplaylists',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('spotify_playlist_id', sa.String(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tplaylists_id'), 'tplaylists', ['id'], unique=False)
    op.create_index(op.f('ix_tplaylists_spotify_playlist_id'), 'tplaylists', ['spotify_playlist_id'], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_tplaylists_spotify_playlist_id'), table_name='tplaylists')
    op.drop_index(op.f('ix_tplaylists_id'), table_name='tplaylists')
    op.drop_table('tplaylists')
    op.drop_index(op.f('ix_users_spotify_id'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_playlist_datasets_playlist_id'), table_name='playlist_datasets')
    op.drop_index(op.f('ix_playlist_datasets_id'), table_name='playlist_datasets')
    op.drop_table('playlist_datasets')
    # ### end Alembic commands ###