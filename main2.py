from flask import Flask, request, render_template, send_from_directory

from rating.pipeline.batch_prediction import PREDICTION_DIR
from rating.utils import load_object
import joblib  # For loading model and transformer
import pandas as pd
from rating.predictor import ModelResolver
from rating.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact
import os,sys
from rating.logger import logging
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    predictions_table = None  # Initialize predictions_table variable
    download_link = None  # Initialize download_link variable
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the uploaded file into a DataFrame
            df = pd.read_csv(uploaded_file)
            df.rename(columns={'listed_in(type)': 'type'}, inplace=True)
            df['type'] = df['type'].apply(lambda x: x.lower())
            df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
            df.drop("reviews_list", axis=1, inplace=True)

            logging.info(f"Creating model resolver object")
            model_resolver = ModelResolver(model_registry="saved_models")
            #logging.info(f"Reading file :{features}")

            logging.info(f"Loading transformer to transform dataset")
            transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
            model = load_object(file_path=model_resolver.get_latest_model_path())
            input_feature_names = list(transformer.feature_names_in_)

            input_arr = transformer.transform(df[input_feature_names])

            prediction = model.predict(input_arr)

            # Create a DataFrame with predictions
            prediction_df = pd.DataFrame({'Predictions': prediction})
            df = pd.concat([df, prediction_df], axis=1)

            # Convert the DataFrame to an HTML table
            predictions_table = df.to_html(classes='table table-bordered table-hover table-responsive')

            # Save predictions to a CSV file
            prediction_csv = 'prediction.csv'
            df.to_csv(prediction_csv, index=False)

            # Set download_link to provide the download button
            download_link = prediction_csv

    return render_template('upload.html', predictions_table=predictions_table, download_link=download_link)

if __name__ == '__main__':
    app.run(debug=True)