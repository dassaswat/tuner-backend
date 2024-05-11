"""Playlist Builder Routes."""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import APIRouter, status, Depends, HTTPException

from .. import schemas, crud
from ..database import get_db
from .. import fix_playlist

router = APIRouter(prefix="/api/v1/tuner", tags=["Playlist Tuner"])


#  response_model=schemas.TunedPlaylist
@router.post("/tune_playlist")
def tune_playlist(data: schemas.TunePlaylist, db: Session = Depends(get_db)):
    """Tune the Playlist"""
    try:
        data.tracks_features = [
            feature for feature in data.tracks_features if feature is not None
        ]
        training_data = crud.get_all_playlist_features(db)
        track_uris, datapoint, weights, evaluation_score = (
            fix_playlist.analyse_and_fix_playlist(training_data, data)
        )
        if isinstance(datapoint, pd.DataFrame):
            crud.create_playlist_feature_datapoint(
                db,
                schemas.PlaylistFeatureCreate(
                    **datapoint.reset_index().to_dict(orient="records")[0]
                ),
            )

            weights_as_list = weights.tolist()
            crud.create_model_weights(
                db,
                schemas.ModelWeightsCreate(
                    energy=weights_as_list[0],
                    liveness=weights_as_list[1],
                    tempo=weights_as_list[2],
                    speechiness=weights_as_list[3],
                    acousticness=weights_as_list[4],
                    instrumentalness=weights_as_list[5],
                    danceability=weights_as_list[6],
                    loudness=weights_as_list[7],
                    evaluation_score=evaluation_score,
                ),
            )

        total_duration = sum(track.duration_ms for track in data.tracks_features)
        return schemas.TunedPlaylist(
            uris=track_uris, length=len(track_uris), duration_ms=total_duration
        )

    except Exception as error:
        print("Error while building playlist", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while building playlist {error}",
        ) from error
