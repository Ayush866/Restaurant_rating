from rating import utils

from rating.entity import config_entity
from rating.entity import artifact_entity
from rating.exception import CustomException
from rating.logger import logging
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20} Data Ingestion {'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            # Exporting collection data as pandas dataframe
            df = pd.read_csv(r"zomato_1.csv")

            logging.info("Save data in feature store")
            df.drop(["url", "phone", "address", "reviews_list", "menu_item"], axis=1, inplace=True)

            df['rate'] = df['rate'].astype(str).apply(utils.fix_rate)
            df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

            df.drop("dish_liked", axis=1, inplace=True)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["Target"] = df["rate"].apply(utils.create_target)
            df.rename(columns={'listed_in(type)': 'type'}, inplace=True)
            df['type'] = df['type'].apply(lambda x: x.lower())

            df_location_count = df["location"].value_counts(normalize=True) * 100
            Desired_index = df_location_count[df_location_count.values > 0.5].index

            def Reduce_Location(r):
                if r in Desired_index:
                    return r
                else:
                    return "other"

            df["location"] = df["location"].apply(Reduce_Location)
            # # Drop unnecessary columns
            df.drop(['name', 'rate', 'cuisines', 'rest_type'], axis=1, inplace=True)
            # Save data in feature store
            logging.info("Create feature store folder if not available")
            # Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info("Save df to feature store folder")
            # Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info("split dataset into train and test set")
            # split dataset into train and test set
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info("create dataset directory folder if not available")
            # create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("Save df to feature store folder")
            # Save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
