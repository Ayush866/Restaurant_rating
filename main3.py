# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle

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
            df.drop("reviews_list",axis=1,inplace=True)

            transformer = pickle.load(open("E:/hotel_rating/saved_models/0/transformer/transformer.pkl", "rb"))
            model = pickle.load(open("E:/hotel_rating/saved_models/0/model/model.pkl", "rb"))

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