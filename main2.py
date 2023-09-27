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
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the uploaded file into a DataFrame
            df = pd.read_csv(uploaded_file)
            df.rename(columns={'listed_in(type)': 'type'}, inplace=True)
            df['type'] = df['type'].apply(lambda x: x.lower())
            df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

            logging.info(f"Creating model resolver object")
            model_resolver = ModelResolver(model_registry="saved_models")
            #logging.info(f"Reading file :{features}")

            logging.info(f"Loading transformer to transform dataset")
            transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
            input_feature_names = list(transformer.feature_names_in_)
            model = load_object(file_path=model_resolver.get_latest_model_path())
            input_arr = transformer.transform(df[input_feature_names])

            prediction = model.predict(input_arr)
            # Apply data transformation using the loaded transformer
            # transformed_data = transformer.transform(input_arr)

            # Make predictions using the loaded model
            # predictions = model.predict(transformed_data)

            # Create a DataFrame with predictions
            prediction_df = pd.DataFrame({'Predictions': prediction})
            df=pd.concat([df,prediction_df],axis=1)
            # Save predictions to a CSV file
            prediction_csv = 'predictions.csv'
            df.to_csv(prediction_csv, index=False)

            # Return the file for download
            return send_from_directory('.', prediction_csv, as_attachment=True)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)