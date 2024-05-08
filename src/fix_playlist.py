"""Fix the playlist"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_data(data):
    """Process the data."""
    df = pd.DataFrame(data)
    df.set_index("id", inplace=True)
    df.drop(
        columns=[
            "uri",
            "type",
            "track_href",
            "analysis_url",
            "duration_ms",
            "time_signature",
            "mode",
        ],
        inplace=True,
    )

    scaler = StandardScaler()
    df[["tempo_t", "key_t", "loudness_t"]] = scaler.fit_transform(
        df[["tempo", "key", "loudness"]]
    )
    return df[
        [
            "energy",
            "liveness",
            "tempo_t",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "danceability",
            "valence",
            "key_t",
            "loudness_t",
        ]
    ]


def construct_distance_matrix(df):
    """Construct the distance matrix."""
    num_songs = df.shape[0]
    distance_matrix = np.zeros((num_songs, num_songs))

    # Single loop impl - (https://jaykmody.com/blog/distance-matrices-with-numpy/)
    # This is fast double loop impl is crazy slow on large datasets
    for i in range(num_songs):
        distance_matrix[i, :] = np.sqrt(np.sum((df.iloc[i] - df) ** 2, axis=1))

    return distance_matrix


def get_minimum_allowed_distance(distance_matrix, percentile):
    """Get the minimum allowed distance."""
    all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    min_allowed_distance = np.percentile(all_distances, percentile)
    return min_allowed_distance


def build_playlist(
    distance_matrix, num_songs, initial_percentile=10, max_relaxations=3
):
    """Build the playlist."""

    playlist = []
    remaining_songs = list(range(num_songs))
    start_song = np.random.choice(remaining_songs)
    playlist.append(start_song)
    remaining_songs.remove(start_song)
    min_allowed_distance = get_minimum_allowed_distance(
        distance_matrix, initial_percentile
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
            if not np.isinf(min_distance):
                next_song = remaining_songs[np.where(distances == min_distance)[0][0]]
                playlist.append(next_song)
                remaining_songs.remove(next_song)

                if relaxation_count:
                    relaxation_count = 0
                    base_relaxation_factor = 0.5
                    min_allowed_distance = get_minimum_allowed_distance(
                        distance_matrix, initial_percentile
                    )

        else:
            if relaxation_count < max_relaxations:
                min_allowed_distance = base_relaxation_factor * min_allowed_distance
                relaxation_count += 1
                base_relaxation_factor = 0.25 * base_relaxation_factor
            else:
                # If we've relaxed the constraints too many times, just add the closest song
                next_song = remaining_songs[np.argmin(distances)]
                playlist.append(next_song)
                remaining_songs.remove(next_song)

    return playlist


def analyse_and_fix_playlist(data):
    """Analyse and fix the playlist."""

    df = process_data(data)
    distance_matrix = construct_distance_matrix(df)
    playlist = build_playlist(distance_matrix, df.shape[0], initial_percentile=10)
    playlist_ids = df.iloc[playlist].index
    playlist_uris = [f"spotify:track:{id}" for id in playlist_ids]

    return playlist_uris
