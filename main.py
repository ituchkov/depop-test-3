import logging
import numpy as np
from feature_extractor import extract_features
from nlp_singleton import NLPModelSingleton
import xgboost as xgb

from sklearn.model_selection import train_test_split
import pandas as pd

import nltk

from config import Classifier, DatasetConfig, FeatureExtractor
from utils import calculate_metrics, insert_named_entities_text, preprocess_descriptions

nltk.download('punkt')
np.random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    nlp_singleton = NLPModelSingleton.get_instance()

    train = pd.read_csv(DatasetConfig.TRAIN_DATA_PAHT, sep=DatasetConfig.DATASET_SEPARATOR)[
        [*DatasetConfig.DATA_COLUMNS]]
    test = pd.read_csv(DatasetConfig.TEST_DATA_PATH, sep=DatasetConfig.DATASET_SEPARATOR)[
        [*DatasetConfig.DATA_COLUMNS]]

    df = pd.concat([train, test], axis=0)
    df = df.reset_index(drop=True)

    df = preprocess_descriptions(df, DatasetConfig.DESCRIPTION_COLUMN_NAME)
    df = insert_named_entities_text(
        df, DatasetConfig.NAMED_ENTITIES_COLUMN_NAME, DatasetConfig.DESCRIPTION_NLP_COLUMN_NAME)
    df = extract_features(df, [DatasetConfig.DESCRIPTION_COLUMN_NAME, DatasetConfig.NAMED_ENTITIES_COLUMN_NAME], [
        FeatureExtractor.DESCRIPTION_FEATURES_NUMBER, FeatureExtractor.NAMED_ENTITIES_FEATURES_NUMBER])
    df = df.drop(
        [DatasetConfig.DESCRIPTION_COLUMN_NAME, DatasetConfig.PRODUCT_ID_COLUMN_NAME, DatasetConfig.DESCRIPTION_TRUECASE_COLUMN_NAME,
            DatasetConfig.DESCRIPTION_NLP_COLUMN_NAME, DatasetConfig.NAMED_ENTITIES_COLUMN_NAME], axis=1
    )

    labels = df[DatasetConfig.LABEL_COLUMN_NAME]
    features = df.drop(DatasetConfig.LABEL_COLUMN_NAME, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=DatasetConfig.DATASET_TEST_SIZE)

    xgb_model = xgb.XGBClassifier(
        learning_rate = Classifier.LEARNING_RATE,
        max_depth = Classifier.MAX_DEPTH,
        n_estimators= Classifier.N_ESTIMATORS,
        subsample = Classifier.SUBSAMPLE)
    
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  eval_metric=Classifier.EVALUATION_METRIC_TYPE, verbose=Classifier.IS_SHOW_TRAINING_LOGS
                  )

    y_pred = xgb_model.predict(X_test)

    calculate_metrics(y_test, y_pred)
