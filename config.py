from dataclasses import dataclass, field
import os
from typing import Dict, Tuple, Union


@dataclass
class ProjectStructure:
    """
    Class representing the project structure configuration.

    Attributes:
    CURRENT_WORKING_DIRECTORY (str): The current working directory.
    DATA_FOLDER (str): The folder where data is stored.
    MODELS_FOLDER (str): The folder where models are stored.
    """
    CURRENT_WORKING_DIRECTORY: str = os.getcwd()
    DATA_FOLDER: str = "data"
    MODELS_FOLDER: str = "models"


@dataclass
class DatasetConfig:
    """
    Class representing the dataset configuration.

    Attributes:
    TEST_DATA_PATH (str): The path to the test data.
    TRAIN_DATA_PAHT (str): The path to the train data.
    DATA_COLUMNS (Tuple[str]): The columns in the dataset.
    DESCRIPTION_COLUMN_NAME (str): The name of the description column.
    PRODUCT_ID_COLUMN_NAME (str): The name of the product ID column.
    DESCRIPTION_TRUECASE_COLUMN_NAME (str): The name of the truecased description column.
    DESCRIPTION_NLP_COLUMN_NAME (str): The name of the NLP processed description column.
    NAMED_ENTITIES_COLUMN_NAME (str): The name of the named entities column.
    LABEL_COLUMN_NAME (str): The name of the label column.
    DATASET_SEPARATOR (str): The separator used in the dataset file.
    DATASET_TEST_SIZE (float): The proportion of the dataset to include in the test split.
    """
    TEST_DATA_PATH: str = f"{ProjectStructure.CURRENT_WORKING_DIRECTORY}/{ProjectStructure.DATA_FOLDER}/test_set.tsv"
    TRAIN_DATA_PAHT: str = f"{ProjectStructure.CURRENT_WORKING_DIRECTORY}/{ProjectStructure.DATA_FOLDER}/train_set.tsv"
    DATA_COLUMNS: Tuple[str] = ("product_id", "description", "label")
    PRODUCT_ID_COLUMN_NAME: str = DATA_COLUMNS[0]
    DESCRIPTION_COLUMN_NAME: str = DATA_COLUMNS[1]
    LABEL_COLUMN_NAME: str = DATA_COLUMNS[2]
    DESCRIPTION_TRUECASE_COLUMN_NAME: str = "description_truecase"
    DESCRIPTION_NLP_COLUMN_NAME: str = "description_nlp"
    NAMED_ENTITIES_COLUMN_NAME: str = "named_entities"
    DATASET_SEPARATOR: str = "\t"
    DATASET_TEST_SIZE: float = 0.25


@dataclass
class FeatureExtractor:
    """
    Class representing the feature extractor configuration.

    Attributes:
    TEXT_LANGUAGE (str): The language of the text data.
    NAMED_ENTITIES_FEATURES_NUMBER (int): The number of named entities features to extract.
    DESCRIPTION_FEATURES_NUMBER (int): The number of description features to extract.
    """
    TEXT_LANGUAGE: str = "english"
    NAMED_ENTITIES_FEATURES_NUMBER: int = 60
    DESCRIPTION_FEATURES_NUMBER: int = 1000


@dataclass
class Classifier:
    """
    Class representing the classifier configuration.

    Attributes:
    PARAMETERS_DICT (dict): The parameters for the classifier.
    EVALUATION_METRIC_TYPE (str): The evaluation metric type.
    IS_SHOW_TRAINING_LOGS (bool): Flag to show training logs.
    """
    LEARNING_RATE = 0.05362886970967467
    MAX_DEPTH = 4
    N_ESTIMATORS =  500
    SUBSAMPLE = 0.9020512553876817
    EVALUATION_METRIC_TYPE: str = "logloss"
    IS_SHOW_TRAINING_LOGS: bool = True
