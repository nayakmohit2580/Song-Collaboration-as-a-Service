import math

import spotipy
import json
import pandas as pd
import random

from inflection import camelize
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

"""
Set the below environment variables to authenticate with spotify.
Get these values from the Spotify Developer Dashboard: https://developer.spotify.com/dashboard/applications
SPOTIPY_CLIENT_ID
SPOTIPY_CLIENT_SECRET
SPOTIPY_REDIRECT_URI
"""


def write_dict_to_json_file(d, file_name, indent=4):
    with open(file_name, 'w+') as file:
        json.dump(d, file, indent=indent)


def list_append(l, item):
    l.append(item)
    return l


def get_track_ids_list(tracks):
    return [track['id'] for track in tracks]


class SpotifyClient:
    feature_keys = [
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms',
        'time_signature',
    ]

    def __init__(self):
        self.client = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    def get_track(self, track_id):
        raw_track_dict = self.client.track(track_id)
        track_dict = {
            'TrackId': track_id,
            'TrackName': raw_track_dict['name'],
            'AlbumId': raw_track_dict['album']['id'],
            'AlbumName': raw_track_dict['album']['name'],
            'Url': raw_track_dict['external_urls']['spotify'],
            'ReleaseDate': raw_track_dict['album']['release_date'],
            'Artists': [{'id': artist['id'], 'name': artist['name']} for artist in raw_track_dict['album']['artists']],
            'Features': self.get_track_features(track_id)
        }

        return track_dict

    def get_track_features_df(self, track_ids, max_tracks=100, include_track_ids=False):
        requests = math.ceil(len(track_ids) / max_tracks)
        data_dict = {key: [] for key in self.feature_keys}
        if include_track_ids:
            data_dict.update({'id': []})
        for i in range(requests):
            start = i * max_tracks
            end = start + max_tracks
            if start == max_tracks:
                return

            if end > len(track_ids):
                end = len(track_ids)
            response = self.client.audio_features(tracks=track_ids[start:end])
            for track in response:
                features = self.filter_track_features(track)
                [data_dict.update({key: list_append(data_dict[key], features[key])}) for key in self.feature_keys]
                if include_track_ids:
                    data_dict.update({'id': list_append(data_dict['id'], track['id'])})

        return pd.DataFrame.from_dict(data_dict, orient='columns')

    def filter_track_features(self, track_response):
        return {key: track_response[key] for key in self.feature_keys}

    def get_track_features(self, track_id):
        response = self.client.audio_features(tracks=[track_id])

        if response:
            return {camelize(key): response[0][key] for key in self.feature_keys}
        else:
            return {}

    def get_playlist_tracks(self, playlist_id):
        response = self.client.playlist_items(playlist_id)

        return [self.get_track(item['track']['id']) for item in response.get('items') if item['track']]

    def search_for_tracks(self, query, limit=10):
        response = self.client.search(query, limit, type='track')

        return [self.get_track(item['id']) for item in response.get('tracks').get('items')]

    def get_recommendations_df(self, seed_track_ids, limit=100, include_track_ids=False):
        response = self.client.recommendations(seed_tracks=seed_track_ids, limit=limit, country='US')
        track_ids = get_track_ids_list(response['tracks'])
        return self.get_track_features_df(track_ids, include_track_ids=include_track_ids)

    def get_random_track_id(self):
        tracks = self.client.recommendations(seed_genres=[self.get_random_genre()], limit=100, country='US')
        return random.choice(tracks['tracks'])['id']

    def get_random_track_ids(self, genre_seeds=None, limit=100):
        if genre_seeds is None:
            genre_seeds = []
        tracks = self.client.recommendations(seed_genres=genre_seeds, limit=limit, country='US')

        return [track['id'] for track in tracks['tracks']]

    def get_random_genre(self):
        return random.choice(self.get_genre_seeds())

    def get_genre_seeds(self):
        return self.client.recommendation_genre_seeds()['genres']

    def get_genre_of_track(self, track_id):
        return self.get_track(track_id)


class SpotifyUserAuthorizedClient(SpotifyClient):
    def __init__(self, scopes=None):
        super().__init__()
        if scopes is None:
            scopes = [
                'user-modify-playback-state',
                'playlist-modify-public',
                'user-library-modify',
                'streaming'
            ]
        self.client = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=" ".join(scopes)))

    def get_current_user_id(self):
        return self.client.current_user().get('id')

    def create_playlist(self, name, user=None):
        if not user:
            user = self.get_current_user_id()
        response = self.client.user_playlist_create(user, name, public=True)

        return response.get('id')

    def delete_playlist(self, playlist_id):
        self.client.current_user_unfollow_playlist(playlist_id)

    def add_track_to_playlist(self, playlist_id, track_id):
        self.client.playlist_add_items(playlist_id, [track_id])

    def remove_track_from_playlist(self, playlist_id, track_id, user=None):
        if not user:
            user = self.get_current_user_id()
        self.client.playlist_remove_all_occurrences_of_items(user, playlist_id, [track_id])

    def follow_playlist(self, playlist_id):
        self.client.current_user_follow_playlist(playlist_id)

    def dislike_track(self, track_id):
        self.client.current_user_saved_tracks_delete([track_id])

    def like_track(self, track_id):
        self.client.current_user_saved_tracks_add([track_id])

    def start_playback(self, playlist_id, track_number=None, position_ms=None):
        try:
            playlist_uri = f'spotify:playlist:{playlist_id}'
            self.client.start_playback(context_uri=playlist_uri, offset=track_number, position_ms=position_ms)
        except spotipy.SpotifyException as e:
            return e

    def pause_playback(self):
        try:
            self.client.pause_playback()
        except spotipy.SpotifyException as e:
            return e
