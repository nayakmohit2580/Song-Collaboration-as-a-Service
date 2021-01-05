from itertools import combinations

import boto3, os, sagemaker, io, datetime, csv, random
from songselection import db, spotifyclient
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.image_uris import retrieve
from sagemaker.predictor import csv_serializer, json_deserializer
import numpy as np
import pandas as pd
import threading
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

retrain_interval = 20
start_train_interval = 10
start_percentage_chance_random = .10
annealing_rate = 0.001


def load_data(room_id, df, validation_size=0.15, test_size=None):
    train, val, test_df = split_data(df, validation_size=validation_size, test_size=test_size)
    upload_data_frame(room_id, train, 'train')
    upload_data_frame(room_id, val, 'validation')
    upload_data_frame(room_id, test_df, 'test')

    return train, val, test_df


def upload_data_frame(room_id, df, key):
    bucket_name = get_bucket_name()
    aws_id, aws_secret = get_auth()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, header=False, index=False)
    s3_resource = boto3.resource('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)
    s3_resource.Object(bucket_name, get_s3_data_path(room_id, key)).put(Body=csv_buffer.getvalue())


def merge_x_y(x, y, target_col_name='Y'):
    df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)
    df.insert(0, 'Y', y_df.pop(target_col_name))

    return df


def split_data(df, validation_size=0.3, test_size=None, target_col_name='Y'):
    Y1 = df.pop(target_col_name)
    X1 = df.copy()
    if not test_size:
        test_size = 0.0

    non_train_size = validation_size + test_size

    X_train, X_val, y_train, y_val = train_test_split(X1, Y1, test_size=non_train_size)

    if test_size > 0:
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=(test_size / non_train_size))
        test_df = merge_x_y(X_test, y_test)
    else:
        test_df = pd.DataFrame()

    train_df = merge_x_y(X_train, y_train)
    val_df = merge_x_y(X_val, y_val)

    return train_df, val_df, test_df


def preprocess_df(df, target_col_name='Y'):
    return move_to_first_col(df, target_col_name)


def preprocess_predict_df(df, target_col_name='Y'):
    df = preprocess_df(df, target_col_name)
    cols_input = list(df.columns)
    cols_input.remove(target_col_name)
    return df[cols_input].values


def move_to_first_col(df, col_name):
    first_col = df.pop(col_name)
    df.insert(0, col_name, first_col)

    return df


def get_image():
    region = os.environ['AWS_DEFAULT_REGION']
    region = region if region else 'us-east-1'

    return retrieve('xgboost', region, version='1.2-1')


def get_minutes_elapsed(start, end):
    return (end - start).total_seconds() / 60


def get_models(room_id):
    return get_s3_objects(f'{room_id}/models')


def get_latest_model_artifact(room_id):
    return f's3://{get_bucket_name()}/{get_latest_model(room_id).key}'


def get_latest_model(room_id):
    models = get_models(room_id)
    latest_model = None
    latest_date_str = ''
    for model in models:
        date_str = str(model.key).replace('sagemaker-xgboost', '')
        if date_str > latest_date_str:
            latest_model = model
            latest_date_str = date_str

    return latest_model


def delete_old_models(room_id):
    latest_model = get_latest_model(room_id)
    models_to_delete = list(get_models(room_id))
    models_to_delete.remove(latest_model)
    [model.delete() for model in models_to_delete]


def get_mme(session=None):
    if not session:
        session = sagemaker.Session()

    return MultiDataModel(name='song-selection-aas-multimodel',
                          model_data_prefix=f's3://{get_bucket_name()}/',
                          sagemaker_session=session,
                          image_uri=get_image(),
                          role=get_sagemaker_role())


def deploy_model(session, room_id):
    mme = get_mme(session=session)
    mme.add_model(get_latest_model_artifact(room_id))


def deploy_multi_model_endpoint(session=None):
    mme = get_mme(session)
    predictor = mme.deploy(initial_instance_count=1,
                           instance_type='ml.m4.xlarge',
                           endpoint_name='song-selection-aas-multimodel')

    return predictor


def get_auth():
    return os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY')


def get_bucket_name():
    return os.environ.get('AWS_S3_BUCKET_NAME')


def get_bucket():
    s3_client = get_s3_client()
    return s3_client.Bucket(get_bucket_name())


def get_sagemaker_role():
    role_name = os.environ.get('AWS_SAGEMAKER_ROLE')
    iam = boto3.client('iam')
    return iam.get_role(RoleName=role_name)['Role']['Arn']


def cleanup_models(room_id):
    try:
        cleanup_start = datetime.datetime.now()
        delete_old_models(room_id)
        cleanup_end = datetime.datetime.now()
        print(f'Cleanup {room_id} took {get_minutes_elapsed(cleanup_start, cleanup_end)} minutes')
    except Exception as e:
        print(f'Failed to cleanup {room_id} models: {e}')


def get_csv_data_s3(room_id, key):
    return sagemaker.session.s3_input("s3://{}/{}/{}".format(get_bucket_name(), room_id, f'{key}.csv'),
                                      content_type='text/csv')


def train_model(room_id):
    bucket_name = get_bucket_name()
    role = get_sagemaker_role()
    s3_input_train = get_csv_data_s3(room_id, 'train')
    s3_input_validate = get_csv_data_s3(room_id, 'validation')
    sess = sagemaker.Session()
    xgb = sagemaker.estimator.Estimator(get_image(),
                                        role,
                                        train_instance_count=1,
                                        train_instance_type='ml.m4.xlarge',
                                        output_path='s3://{}/{}/models'.format(bucket_name, room_id),
                                        sagemaker_session=sess,
                                        base_job_name='sagemaker-xgboost')
    xgb.set_hyperparameters(objective='reg:linear', num_round=25)

    try:
        train_start = datetime.datetime.now()
        xgb.fit({'train': s3_input_train, 'validation': s3_input_validate})
        train_end = datetime.datetime.now()
        print(f'Training {room_id} took {get_minutes_elapsed(train_start, train_end)} minutes')
    except Exception as e:
        print(f'Failed to train {room_id}: {e}')
        return

    try:
        deploy_start = datetime.datetime.now()
        deploy_model(sess, room_id)
        deploy_end = datetime.datetime.now()
        print(f'Deployment {room_id} took {get_minutes_elapsed(deploy_start, deploy_end)} minutes')
    except Exception as e:
        print(f'Failed to deploy {room_id}: {e}')
        return


def get_s3_data_path(room_id, data_type):
    return f'{room_id}/{data_type}.csv'


def get_s3_client():
    aws_id, aws_secret = get_auth()
    return boto3.resource('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)


def get_s3_objects(prefix):
    bucket_name = get_bucket_name()
    s3_resource = get_s3_client()
    bucket = s3_resource.Bucket(bucket_name)
    return bucket.objects.filter(Prefix=prefix)


def model_exists(room_id):
    objects = list(get_s3_objects(f'{room_id}/models'))
    if objects:
        return True
    else:
        return False


def delete_room(room_id):
    get_s3_objects(room_id).delete()


def predict(room_id, df=None):
    try:
        predict_start = datetime.datetime.now()
        predictor = sagemaker.predictor.RealTimePredictor('song-selection-aas-multimodel', serializer=csv_serializer,
                                                          deserializer=json_deserializer, content_type='text/csv')
        if df is not None:
            data = preprocess_predict_df(df)
        else:
            df = get_s3_dataframe(room_id, 'test')
            data = preprocess_predict_df(df, df.columns[0])
        model = get_latest_model(room_id).key
        predicted_values = predictor.predict(data, target_model=model)
        predict_end = datetime.datetime.now()
        print(f'Prediction {room_id} took {get_minutes_elapsed(predict_start, predict_end)} minutes')

        return predicted_values
    except Exception as e:
        print(f'Failed to predict {room_id}: {e}')


def net_reaction(up, down):
    return up - down


def get_room_tracks(room_id):
    cur = db.connection.cursor()
    query = f'SELECT TrackID, LikeCount, DislikeCount FROM Reactions WHERE RoomID="{room_id}"'
    cur.execute(query)
    rows = cur.fetchall()
    records = [{'id': row[0], 'Y': net_reaction(row[1], row[2])} for row in rows]
    df = pd.DataFrame.from_records(records)
    db.connection.commit()

    return df


def get_track_pool(tracks, random_track=False):
    client = spotifyclient.SpotifyClient()
    n = 4 if random_track else 5
    df = tracks.nlargest(n, 'Y')

    if len(tracks.index) <= start_train_interval:
        df = df.loc[df['Y'] > 0]
        if len(df.index) == 0:
            df = tracks.nlargest(1, 'Y')

    top_seeds = list(df['id'])
    if random_track:
        top_seeds.append(client.get_random_track_id())

    return client.get_recommendations_df(top_seeds, include_track_ids=True)


def get_spotify_recommendation(room_id):
    tracks = get_room_tracks(room_id)
    track_pool = get_track_pool(tracks)
    recommendation = list(track_pool.sample()['id'].values)[0]
    return recommendation


def get_random_recommendation():
    return spotifyclient.SpotifyClient().get_random_track_id()


def join_reactions_and_features(tracks):
    client = spotifyclient.SpotifyClient()
    tracks.id.astype(object)
    df = client.get_track_features_df(tracks['id'], include_track_ids=True)
    df.id.astype(object)

    return pd.merge(df, tracks, on='id', how='left')


def upload_and_train_model(room_id, df):
    load_data(room_id, df)
    train_model(room_id)


def get_recommendation_from_genre(genre):
    return random.choice(spotifyclient.SpotifyClient().get_random_track_ids(genre_seeds=[genre]))


def get_recommendation(room_id, track_number):
    if track_number == start_train_interval or (track_number > start_train_interval and track_number % retrain_interval == 0):
        tracks = get_room_tracks(room_id)
        df = join_reactions_and_features(tracks)
        df.pop('id')
        thread = threading.Thread(target=upload_and_train_model, args=[room_id, df])
        thread.daemon = True
        thread.start()

    percentage_chance_random = start_percentage_chance_random - (track_number * annealing_rate)
    if percentage_chance_random <= 0:
        percentage_chance_random = annealing_rate

    if random.random() < percentage_chance_random:
        rec = get_random_recommendation()
    elif model_exists(room_id):
        rec = get_max_track_id(room_id)
    else:
        rec = get_spotify_recommendation(room_id)

    return rec


def get_random_data(cnt=100):
    client = spotifyclient.SpotifyClient()
    track_ids = [client.get_random_track_id() for i in range(cnt)]
    feature_df = client.get_track_features_df(track_ids)
    feature_df['Y'] = np.random.randint(-5, 5, feature_df.shape[0])

    return feature_df


def random_data_model(room_id, cnt=100):
    load_data(room_id, get_random_data(cnt))
    train_model(room_id)


def get_s3_dataframe(room_id, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=get_bucket_name(), Key=get_s3_data_path(room_id, key))
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def synthetic_data_test(genre1=None, genre2=None, instance_number=100):
    client = spotifyclient.SpotifyClient()
    if not genre1:
        genre1 = client.get_random_genre()
    if not genre2:
        genre2 = client.get_random_genre()
    room_id = f'{genre1}-{genre2}-test'
    genre1_df = client.get_track_features_df(client.get_random_track_ids(genre_seeds=[genre1], limit=instance_number))
    genre2_df = client.get_track_features_df(client.get_random_track_ids(genre_seeds=[genre2], limit=instance_number))
    genre1_df['Y'] = 0
    genre2_df['Y'] = 1
    df = genre1_df.append(genre2_df)
    df.sample(frac=1)
    load_data(room_id, df, 0.15, 0.15)
    train_model(room_id)
    train_df = get_s3_dataframe(room_id, 'test')
    predictions = predict(room_id)
    actual = list(train_df.iloc[:, 0])

    return evaluate(genre1, genre2, actual, predictions)


def get_max_track_id(room_id):
    tracks = get_room_tracks(room_id)
    input_df = get_track_pool(tracks, random_track=True)
    input_df['Y'] = 0
    track_ids = list(input_df.pop('id'))
    predictions = list(predict(room_id, input_df))
    index = predictions.index(max(predictions))

    return track_ids[index]


def evaluate(genre1, genre2, actual, predicted):
    return {
        'genre1': genre1,
        'genre2': genre2,
        'mean_absolute_error': round(sm.mean_absolute_error(actual, predicted), 2),
        'mean_squared_error': round(sm.mean_squared_error(actual, predicted), 2),
        'median_absolute_error': round(sm.mean_absolute_error(actual, predicted), 2),
        'explained_variance_score': round(sm.explained_variance_score(actual, predicted), 2),
        'r2_score': round(sm.r2_score(actual, predicted), 2),
    }


def test():
    try:
        deploy_multi_model_endpoint()
    except Exception as e:
        print(f'Failed to deploy multimodel endpoint: {e}')
    genres = ['jazz', 'rock', 'pop', 'classical']
    results = []
    for combination in combinations(genres, 2):
        results.append(synthetic_data_test(combination[0], combination[1]))

    with open('results.csv', 'w+') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

