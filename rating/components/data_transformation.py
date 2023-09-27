from rating.entity import artifact_entity, config_entity
from rating.exception import CustomException
from rating.logger import logging
from typing import Optional
import os, sys
from sklearn.pipeline import Pipeline
import pandas as pd
from rating import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from rating.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
from category_encoders import BinaryEncoder
from xgboost import XGBClassifier


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Transformation {'<<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            Encoder = ColumnTransformer(transformers=[
                ("OHE", OneHotEncoder(sparse_output=False, drop="first"), ["online_order", "book_table", "type"]),
                ("BE", BinaryEncoder(), ["location", "listed_in(city)"])], remainder="passthrough")

            steps = []

            steps.append(("Encoder", Encoder))
            steps.append(("Scaler", RobustScaler()))
            pipeline = Pipeline(steps=steps)

            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, ) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            # transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)

            input_feature_train_arr1 = input_feature_train_arr
            input_feature_test_arr1 = input_feature_test_arr

            # target_feature_train_arr1 = target_feature_train_df.reshape(-1, 1)
            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values
            # logging.info(f"{target_feature_train_arr1.shape}")
            # target_feature_test_arr1 = target_feature_test_df.reshape(-1, 1)

            test_arr = np.hstack((input_feature_test_arr1, target_feature_test_arr.reshape(-1, 1)))
            train_arr = np.hstack((input_feature_train_arr1, target_feature_train_arr.reshape(-1, 1)))

            # save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipleine)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)