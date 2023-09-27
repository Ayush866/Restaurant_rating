import pandas as pd
from rating.logger import logging
from rating.exception import CustomException
from rating.config import mongo_client
import os, sys
import yaml
import dill
import numpy as np


def get_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id", axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def write_yaml_file(file_path, data: dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise CustomException(e, sys)


def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


class CustomData:
    def __init__(self,
                     age: float,
                     workclass: str,
                     education_num: int,
                     marital_status: str,
                     occupation: str,
                     relationship: str,
                     race: str,
                     sex: str,
                     capital_gain: int,
                     capital_loss: int,
                     hours_per_week: int,country: str):
            self.age = age

            self.workclass = workclass

            self.education_num = education_num

            self.marital_status = marital_status

            self.occupation = occupation

            self.relationship = relationship

            self.race = race

            self.sex = sex

            self.capital_gain = capital_gain

            self.capital_loss = capital_loss

            self.hours_per_week = hours_per_week

            self.country = country

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education-num": [self.education_num],
                "marital-status": [self.marital_status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "sex": [self.sex],
                "capital-gain": [self.capital_gain],
                "capital-loss": [self.capital_loss],
                "hours-per-week": [self.hours_per_week],
                "country": [self.country],

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


def fix_rate(r):
    try:
        if "/" in r:
            return float(r[0:3])
        else:
            return np.nan
    except Exception as e:
        raise CustomException

def create_target(r):
    if r >= 3.75:
        return 1
    else :
        return 0

