"""playlists-features

Revision ID: 1baac5003eed
Revises: bae0a50f8bb6
Create Date: 2024-05-11 00:07:43.089598

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1baac5003eed'
down_revision: Union[str, None] = 'bae0a50f8bb6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('playlists_features',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('playlist_id', sa.String(), nullable=False),
    sa.Column('energy', sa.Numeric(), nullable=False),
    sa.Column('liveness', sa.Numeric(), nullable=False),
    sa.Column('tempo', sa.Numeric(), nullable=False),
    sa.Column('speechiness', sa.Numeric(), nullable=False),
    sa.Column('acousticness', sa.Numeric(), nullable=False),
    sa.Column('instrumentalness', sa.Numeric(), nullable=False),
    sa.Column('danceability', sa.Numeric(), nullable=False),
    sa.Column('loudness', sa.Numeric(), nullable=False),
    sa.Column('valence', sa.Numeric(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_playlists_features_id'), 'playlists_features', ['id'], unique=False)
    op.create_index(op.f('ix_playlists_features_playlist_id'), 'playlists_features', ['playlist_id'], unique=True)
    op.drop_index('ix_playlist_datasets_id', table_name='playlist_datasets')
    op.drop_index('ix_playlist_datasets_playlist_id', table_name='playlist_datasets')
    op.drop_table('playlist_datasets')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('playlist_datasets',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('playlist_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('energy', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('liveness', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('tempo', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('speechiness', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('acousticness', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('instrumentalness', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('danceability', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('loudness', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('valence', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name='playlist_datasets_pkey')
    )
    op.create_index('ix_playlist_datasets_playlist_id', 'playlist_datasets', ['playlist_id'], unique=True)
    op.create_index('ix_playlist_datasets_id', 'playlist_datasets', ['id'], unique=False)
    op.drop_index(op.f('ix_playlists_features_playlist_id'), table_name='playlists_features')
    op.drop_index(op.f('ix_playlists_features_id'), table_name='playlists_features')
    op.drop_table('playlists_features')
    # ### end Alembic commands ###
