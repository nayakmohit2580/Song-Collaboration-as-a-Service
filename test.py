import flask_mysql
import time

from songselection import spotifyclient


def flim_test():
    user_account = spotifyclient.SpotifyUserAuthorizedClient()
    playlist_id = user_account.create_playlist('FLIM-TEST')
    service_account = spotifyclient.SpotifyClient()
    tracks = service_account.search_for_tracks("Flim")
    user_account.add_track_to_playlist(playlist_id, tracks[0]['track_id'])
    playlist_tracks = service_account.get_playlist_tracks(playlist_id)
    spotifyclient.write_dict_to_json_file(playlist_tracks[0], "flim.json")
    user_account.follow_playlist(playlist_id)
    user_account.start_playback(playlist_id)
    time.sleep(10)
    user_account.like_track(playlist_tracks[0]['track_id'])
    user_account.pause_playback()
    user_account.delete_playlist(playlist_id)