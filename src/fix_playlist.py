"""Fix the playlist"""

from typing import Optional

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from . import schemas


def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the data."""

    df.set_index("id", inplace=True)
    df.drop(
        columns=[
            "uri",
            "type",
            "track_href",
            "analysis_url",
            "duration_ms",
            "mode",
            "time_signature",
            "key",
        ],
        inplace=True,
    )

    scaler = MinMaxScaler()
    df[
        [
            "energy",
            "liveness",
            "tempo",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "danceability",
            "valence",
            "loudness",
        ]
    ] = scaler.fit_transform(
        df[
            [
                "energy",
                "liveness",
                "tempo",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "danceability",
                "valence",
                "loudness",
            ]
        ]
    )

    # Get the avearages of the features, for which data is normally distributed
    features_to_calculate_avg_for = df.drop(
        columns=[
            "spotify_playlist_id",
            "speechiness",
            "acousticness",
            "instrumentalness",
        ]
    )

    # Get the median of the features, for which data is not normally distributed
    features_to_calculate_median_for = df.drop(
        columns=[
            "valence",
            "energy",
            "liveness",
            "tempo",
            "danceability",
            "loudness",
            "spotify_playlist_id",
        ]
    )

    avg_features = features_to_calculate_avg_for.mean()
    median_features = features_to_calculate_median_for.median()

    for features in median_features.index:
        avg_features[features] = median_features[features]

    # Convert the series to a dataframe
    data_point = avg_features.to_frame().T
    data_point["spotify_playlist_id"] = df["spotify_playlist_id"].iloc[0]
    data_point.set_index("spotify_playlist_id", inplace=True)

    features = df.drop(columns=["spotify_playlist_id"])
    return (
        data_point,
        features[
            [
                "energy",
                "liveness",
                "tempo",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "danceability",
                "loudness",
                "valence",
            ]
        ],
    )


def create_test_train_split(x, y):
    """Create a test and train split"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.50, random_state=42
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train) -> RandomForestRegressor:
    """Train a Random forest regression model"""
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, x_test, y_test) -> float:
    """Evaluate the model"""
    return model.score(x_test, y_test)


def get_model_feature_importance(model: RandomForestRegressor, x_train) -> pd.DataFrame:
    """Get the feature importance of the model as a dataframe"""
    return pd.DataFrame(
        model.feature_importances_, index=x_train.columns, columns=["Importance"]
    ).sort_values(by="Importance", ascending=False)


def construct_distance_matrix(
    df: pd.DataFrame, weights: np.ndarray | None = None
) -> np.ndarray:
    """Construct the distance matrix."""
    num_songs = df.shape[0]
    distance_matrix = np.zeros((num_songs, num_songs))

    if weights is None:
        weights = np.ones(df.shape[1])
    # Single loop impl - (https://jaykmody.com/blog/distance-matrices-with-numpy/)
    # This is fast, double loop impl is crazy slow on large datasets
    for i in range(num_songs):
        distance_matrix[i, :] = np.sqrt(
            np.sum(((df.iloc[i] - df) ** 2) * weights, axis=1)
        )

    return distance_matrix


def get_emotion_cluster(df: pd.DataFrame, k=4) -> np.ndarray:
    """Get the emotion cluster."""
    x = df["valence"].values.reshape(-1, 1)
    k_means_model = KMeans(n_clusters=k)
    predictions = k_means_model.fit_predict(x)
    return predictions


def find_elbow_point(df: pd.DataFrame, k_max=10):
    """Find the elbow point."""
    means = []
    inertias = []
    data = df["valence"].values.reshape(-1, 1)

    for k in range(1, k_max):
        model = KMeans(n_clusters=k)
        model.fit(data)
        means.append(k)
        inertias.append(model.inertia_)

    kneedle = KneeLocator(means, inertias, curve="convex", direction="decreasing")
    return kneedle.elbow


def get_minimum_allowed_distance(distance_matrix, percentile):
    """Get the minimum allowed distance."""
    all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    min_allowed_distance = np.percentile(all_distances, percentile)
    return min_allowed_distance


def build_playlist(
    distance_matrix,
    emotion_cluster=None,
    seed_tracks: Optional[list[int]] = None,
    thresshold_percentile=10,
    max_relaxations=3,
):
    """Build the playlist."""

    playlist = []
    remaining_songs = list(range(distance_matrix.shape[0]))

    seed_track = (
        np.random.choice(remaining_songs)
        if seed_tracks is None
        else np.random.choice(seed_tracks)
    )

    playlist.append(seed_track)
    remaining_songs.remove(seed_track)
    min_allowed_distance = get_minimum_allowed_distance(
        distance_matrix, thresshold_percentile
    )
    relaxation_count = 0
    base_relaxation_factor = 0.5

    while remaining_songs:
        distances = np.array(
            [distance_matrix[playlist[-1], song] for song in remaining_songs]
        )
        valid_distances = distances[distances >= min_allowed_distance]

        if valid_distances.size:
            min_distance = np.min(valid_distances)

            candidate_distances = [
                distance
                for distance in valid_distances
                if emotion_cluster[playlist[-1]]
                != emotion_cluster[
                    remaining_songs[np.where(distances == distance)[0][0]]
                ]
            ]

            if candidate_distances:
                min_distance = np.min(candidate_distances)

            if not np.isinf(min_distance):
                next_song = remaining_songs[np.where(distances == min_distance)[0][0]]
                playlist.append(next_song)
                remaining_songs.remove(next_song)

                if relaxation_count:
                    relaxation_count = 0
                    base_relaxation_factor = 0.5
                    min_allowed_distance = get_minimum_allowed_distance(
                        distance_matrix, thresshold_percentile
                    )

        else:
            if relaxation_count < max_relaxations:
                min_allowed_distance = base_relaxation_factor * min_allowed_distance
                relaxation_count += 1
                base_relaxation_factor = 0.25 * base_relaxation_factor
            else:
                next_song = remaining_songs[np.argmin(distances)]
                playlist.append(next_song)
                remaining_songs.remove(next_song)

    return playlist


def analyse_and_fix_playlist(training_data, playlist_data):
    """Analyse and fix the playlist."""

    _training_data = [
        schemas.PlaylistFeature.model_validate(data).model_dump()
        for data in training_data
    ]
    training_data_df = pd.DataFrame(_training_data)
    training_data_df.set_index("spotify_playlist_id", inplace=True)
    training_data_df.drop(columns=["id"], inplace=True)

    playlist_data_df = pd.json_normalize(
        playlist_data.model_dump(),
        record_path=["tracks_features"],
        meta=["spotify_playlist_id"],
    )

    # Check if the playlist_data_df spotify_playlist_id is in the training data
    # if not, add it to the training data
    if playlist_data_df.iloc[0]["spotify_playlist_id"] in training_data_df.index:
        print("Playlist feature already exists, skipping training...")
        playlist_data.must_train = False

    datapoint, features = process_data(playlist_data_df)
    weights = np.array(
        [
            0.17324821444202046,
            0.07176071971418228,
            0.09139859881649352,
            0.08270437843029588,
            0.05826917930667714,
            0.37975134149673134,
            0.0652021402236166,
            0.07766542756998279,
        ]
    )
    return_datapoint = False
    feature_importance = None
    evaluation_score = None

    if playlist_data.must_train:
        return_datapoint = True
        print("Training the model...")
        training_data_df = pd.concat([training_data_df, datapoint])
        x = training_data_df.drop(columns=["valence"])
        y = training_data_df["valence"]
        x_train, x_test, y_train, y_test = create_test_train_split(x, y)
        model = train_model(x_train, y_train)
        evaluation_score = evaluate_model(model, x_test, y_test)
        print("Current model evaluation score: ", evaluation_score)
        feature_importance = get_model_feature_importance(model, x_train)
        weights = [
            # pylint: disable=unsubscriptable-object
            feature_importance["Importance"][feature]
            for feature in x_train.columns
        ]
        print("Done training the model...")

    matrix_features = features.drop(columns=["valence"])
    print(matrix_features.columns)
    distance_matrix = construct_distance_matrix(matrix_features, weights)
    elbow_point = find_elbow_point(features)
    emotion_cluster = get_emotion_cluster(features, elbow_point + 1)
    playlist = build_playlist(
        distance_matrix, emotion_cluster, playlist_data.seed_tracks, 20
    )

    playlist_ids = features.iloc[playlist].index
    playlist_uris = [f"spotify:track:{id}" for id in playlist_ids]
    return (
        playlist_uris,
        datapoint if return_datapoint else None,
        weights if return_datapoint else None,
        evaluation_score if return_datapoint else None,
    )
