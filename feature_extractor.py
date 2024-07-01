from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import FeatureExtractor


def extract_features(df: pd.DataFrame, feature_column_names: List[str], feature_lengths: List[int]) -> pd.DataFrame:
    """
    Extract features from text columns in a DataFrame using TF-IDF vectorization.

    Args:
    df (pd.DataFrame): The input DataFrame.
    feature_column_names (List[str]): The names of the columns to extract features from.
    feature_lengths (List[int]): The number of features to extract for each column.

    Returns:
    pd.DataFrame: The DataFrame with extracted features.
    """
    for feature_column_name, feature_length in zip(feature_column_names, feature_lengths):
        feature_vectorizer = TfidfVectorizer(
            stop_words=FeatureExtractor.TEXT_LANGUAGE, max_features=feature_length
        )

        df_tfidf_features = pd.DataFrame(
            feature_vectorizer.fit_transform(
                df[feature_column_name]).todense(),
            columns=feature_vectorizer.get_feature_names_out(),
        )
        df_tfidf_features.columns = [
            f"{i}_{feature_column_name}" for i in df_tfidf_features.columns
        ]

        df = pd.concat([df, df_tfidf_features], axis=1)
    return df