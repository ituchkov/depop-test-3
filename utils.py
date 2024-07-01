
import logging
import pandas as pd
import unidecode
import re
import numpy as np
import truecase

from config import DatasetConfig
from nlp_singleton import NLPModelSingleton
from itertools import chain

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def proportion_of_stopwords(description: str) -> float:
    """
    Calculate the proportion of stopwords in a given description.

    Args:
    description (str): The input text description.

    Returns:
    float: The proportion of stopwords in the description.
    """
    nlp_singleton = NLPModelSingleton.get_instance()

    tokens = description.split(" ")
    num_stopwords = len(
        [word for word in tokens if word.lower(
        ) in nlp_singleton.nlp.Defaults.stop_words]
    )
    return float(num_stopwords) / float(len(tokens)) if len(tokens) else 0.0


def average_length_of_word(description: str) -> float:
    """
    Calculate the average length of words in a given description.

    Args:
    description (str): The input text description.

    Returns:
    float: The average length of words in the description.
    """
    tokens = description.split(" ")
    return np.mean([len(word) for word in tokens]) if len(tokens) else 0.0


def proportion_of_numbers(description: str) -> float:
    """
    Calculate the proportion of numerical tokens in a given description.

    Args:
    description (str): The input text description.

    Returns:
    float: The proportion of numerical tokens in the description.
    """
    tokens = description.split(" ")
    num_digits = len([word for word in tokens if word.isdigit()])
    return float(num_digits) / float(len(tokens)) if len(tokens) else 0.0


def normalise_nonascii_chars(input_str: str) -> str:
    """
    Normalize non-ASCII characters in a string to their closest ASCII equivalent.

    Args:
    input_str (str): The input string.

    Returns:
    str: The normalized string.
    """
    return unidecode.unidecode(input_str)


def replace_special_chars(main_string: str) -> str:
    """
    Replace special characters in a string with spaces.

    Args:
    main_string (str): The input string.

    Returns:
    str: The string with special characters replaced by spaces.
    """
    return re.sub("[,;@#!\?\+\*\n\-: /]", " ", main_string)


def keep_alphanumeric_chars(string_input: str) -> str:
    """
    Keep only alphanumeric characters and spaces in a string.

    Args:
    string_input (str): The input string.

    Returns:
    str: The cleaned string with only alphanumeric characters and spaces.
    """
    return re.sub("[^A-Za-z0-9& ]", "", string_input)


def remove_spaces(string_input: str) -> str:
    """
    Remove extra spaces from a string.

    Args:
    string_input (str): The input string.

    Returns:
    str: The string with extra spaces removed.
    """
    return " ".join(string_input.split())


def lemmatize(string_input: str) -> str:
    """
    Lemmatize the input string.

    Args:
    string_input (str): The input string.

    Returns:
    str: The lemmatized string.
    """
    nlp_singleton = NLPModelSingleton.get_instance()
    token_object = nlp_singleton(string_input)
    lemmas_list = [
        word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in token_object
    ]
    return " ".join(lemmas_list)


def clean_description(input_str: str) -> str:
    """
    Clean the input description by normalizing, lemmatizing, and removing special characters.

    Args:
    input_str (str): The input description.

    Returns:
    str: The cleaned description.
    """
    input_str = replace_special_chars(input_str.lower())
    input_str = normalise_nonascii_chars(input_str)
    input_str = keep_alphanumeric_chars(input_str)
    input_str = lemmatize(input_str)
    input_str = remove_spaces(input_str)
    return input_str


def drop_digits(s: str) -> str:
    """
    Remove digits from the input string.

    Args:
    s (str): The input string.

    Returns:
    str: The string with digits removed.
    """
    return "".join([i for i in s if not i.isdigit()])


def preprocess_descriptions(df: pd.DataFrame, preprocess_column_name: str) -> pd.DataFrame:
    """
    Preprocess the descriptions in a DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    preprocess_column_name (str): The name of the column containing descriptions to preprocess.

    Returns:
    pd.DataFrame: The DataFrame with preprocessed descriptions.
    """
    assert preprocess_column_name in df.columns, f"Column '{preprocess_column_name}' not found in DataFrame."
    logging.info(f"Starting preprocessing for column: {preprocess_column_name}")

    nlp_singleton = NLPModelSingleton.get_instance()

    df[preprocess_column_name] = df[preprocess_column_name].apply(
        clean_description)

    pipeline_steps_map = [
        (proportion_of_stopwords, 'proportion_of_stopwords'),
        (average_length_of_word, 'average_length_of_word'),
        (proportion_of_numbers, 'proportion_of_numbers')
    ]

    for func, col_name in pipeline_steps_map:
        df[col_name] = df[preprocess_column_name].apply(func)
        logging.info(f"Applied {func.__name__} and added column: {col_name}")

    df[DatasetConfig.DESCRIPTION_COLUMN_NAME] = df[DatasetConfig.DESCRIPTION_COLUMN_NAME].apply(
        drop_digits)
    df["description_truecase"] = df[DatasetConfig.DESCRIPTION_COLUMN_NAME].apply(
        truecase.get_true_case)
    df["description_nlp"] = df["description_truecase"].apply(nlp_singleton)
    logging.info("Preprocessing completed.")

    return df


def insert_named_entities_text(df: pd.DataFrame, named_entities_column_name: str, description_nlp_column_name: str) -> pd.DataFrame:
    """
    Insert named entities into a DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    named_entities_column_name (str): The name of the column to store named entities.
    description_nlp_column_name (str): The name of the column that will contain NLP-processed descriptions.

    Returns:
    pd.DataFrame: The DataFrame with named entities inserted.
    """
    assert description_nlp_column_name in df.columns, f"Column '{description_nlp_column_name}' not found in DataFrame."

    df[named_entities_column_name] = ""
    logging.info("Starting named entity insertion.")

    for i, description_nlp in df[description_nlp_column_name].items():
        named_entities_sets = description_nlp.ents
        named_entities = list(set(chain(*named_entities_sets)))
        df.loc[i, DatasetConfig.NAMED_ENTITIES_COLUMN_NAME] = " ".join(
            j.text for j in named_entities)
        logging.debug(f"Inserted named entities for row {i}.")
        
    logging.info("Named entity insertion completed.")
    return df


def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Calculate and print classification metrics.

    Args:
    y_test (np.ArrayLike): The true labels.
    y_pred (np.ArrayLike): The predicted labels.

    Returns:
    None
    """
    assert len(y_test) == len(y_pred), "Length of y_test and y_pred must be equal."

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"---Metrics---")
    logging.info(f"Accuracy: {(accuracy * 100.0):.2f}%")
    logging.info(f"Precision: {(precision * 100.0):.2f}%")
    logging.info(f"Recall: {(recall * 100.0):.2f}%")
    logging.info(f"F1 Score: {(f1 * 100.0):.2f}%")
