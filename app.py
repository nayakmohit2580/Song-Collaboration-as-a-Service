from songselection import app, recommendations

if __name__ == '__main__':
    try:
        recommendations.deploy_multi_model_endpoint()
    except Exception as e:
        print(f'Failed to deploy multimodel endpoint: {e}')
    app.run(host='0.0.0.0',port=8080,debug=True)