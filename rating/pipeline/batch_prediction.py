from rating.exception import CustomException
from rating.logger import logging
from rating.predictor import ModelResolver
import pandas as pd
from rating.utils import load_object
import os, sys
from datetime import datetime

PREDICTION_DIR = "prediction"

import numpy as np


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.rename(columns={'listed_in(type)': 'type'}, inplace=True)
        df['type'] = df['type'].apply(lambda x: x.lower())
        df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

        # validation

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_names = list(transformer.feature_names_in_)
        print(input_feature_names)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        # cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"] = prediction
        # df["cat_pred"] = converted_labels

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",
                                                                         f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
    except Exception as e:
        raise CustomException(e, sys)