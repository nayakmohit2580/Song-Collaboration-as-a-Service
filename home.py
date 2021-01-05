from songselection import app, recommendations
import os

if __name__ == '__main__':
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = 'AKIAJ4V2GWH6322HBY2A'
        os.environ["AWS_SECRET_ACCESS_KEY"] = '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'
        os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
        os.environ["AWS_S3_BUCKET_NAME"] = 'sagemaker-song-selection-aas'
        os.environ["AWS_SAGEMAKER_ROLE"] = 'song-selection-aas'
        recommendations.deploy_multi_model_endpoint()
    except Exception as e:
        print(f'Failed to deploy multimodel endpoint: {e}')
    app.run(debug=True)