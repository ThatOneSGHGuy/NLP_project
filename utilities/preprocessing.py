########################################
# Preprocessing steps - helper functions
########################################

import re

import pandas as pd
import unicodedata
from langdetect import detect
from loguru import logger
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandarallel import pandarallel


def get_wordnet_pos_func(
        word: str,
) -> str:
    """
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet

    Args:
        word (str): Word to which the function is to be applied, string

    Returns:
        POS tag, readable for the lemmatizer of wordnet
    """

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def norm_lemm_POS_tag(
        text: str,
) -> str:
    """
    Lemmatize tokens from string

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is determined with the help of function get_wordnet_pos()

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        String with lemmatized words
    """

    words = word_tokenize(text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words])
    return text


def remove_accented_chars(
        text: str,
) -> str:
    """
    Removes all accented characters from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without accented characters
    """

    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_extra_whitespaces(
        text: str,
) -> str:
    """
    Removes extra whitespaces from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without extra whitespaces
    """

    return re.sub(r'^\s*|\s+', ' ', text).strip()


def is_english(
        text: str,
) -> bool:
    """
    Checks if a text is in English using language detection.

    Args:
        text (str): The text to be checked for language.

    Returns:
        bool: True if the text is in English, False otherwise.
    """

    try:
        return detect(text) == 'en'
    except:
        return False


def remove_short_text(
        df: pd.Series,
        text_col: str,
        threshold: int,
) -> pd.Series:
    """
    Removes short text observations from a DataFrame based on a word count threshold.

    Args:
        df (pd.Series): The DataFrame containing text data.
        text_col (str): The name of the column in df that contains the text data.
        threshold (int): The minimum word count threshold for retaining observations.

    Returns:
        pd.Series: The DataFrame with short text observations removed.
    """

    # Create a copy of the DataFrame
    df = df.copy(deep=True)

    # Calculate the length of each text in terms of word count
    len_review = df[text_col].str.split().str.len()

    # Filter and keep only the observations with a word count greater than the threshold
    df = df[len_review > threshold]

    return df


def preprocess_text(
        text: str,
) -> str:
    """
    Preprocesses a text by lowercasing, removing extra whitespaces, tokenizing,
    removing stopwords, removing punctuations, and lemmatizing.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text in form of tokens.
    """

    # Lemmatization
    text = norm_lemm_POS_tag(text)

    # Combine multiple preprocessing steps into one pass
    text = text.lower()  # Lowercasing
    text = remove_accented_chars(text)  # Removing accented characters
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Removing punctuation and numbers
    text = remove_extra_whitespaces(text)  # Removing extra whitespaces

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stop-words
    stopwords_en = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stopwords_en]

    return tokens


def preprocess_texts_eng(
        df: pd.DataFrame,
        text_col: str,
        shuffle: bool,
        train_subset: float = 1,
        workers: int = 1,
) -> pd.DataFrame:
    """
    Preprocesses a DataFrame containing text data by removing short texts, optionally shuffling the data,
    filtering by language, and tokenizing the text.

    Args:
        df (pd.DataFrame): The raw DataFrame containing text data.
        text_col (str): The name of the column in df_raw that contains the text data.
        shuffle (bool): Whether to shuffle the data.
        train_subset (float, optional): Fraction of the data to keep, a value between 0 and 1.
        workers (int): Number of CPUs to use in the text preprocessing.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with an added 'tokens' column containing tokenized text.
    """

    # Enable progress bar
    pandarallel.initialize(nb_workers=workers, progress_bar=True)

    # Create a new df
    data_processed = df.copy(deep=True)

    # Remove observations with less than 3 words in the ReviewText
    data_processed = remove_short_text(df=data_processed, text_col=text_col, threshold=3)

    # Shuffle the data if requested
    if shuffle:
        data_processed = data_processed.sample(frac=1).reset_index(drop=True)

    # Subset the data if train_subset is not 1
    if train_subset != 1:
        data_processed = data_processed[: round(data_processed.shape[0] * train_subset)]

    # Remove any non-English observations
    logger.info("Removing non-English observations")
    non_english_reviews_count = data_processed.shape[0]
    data_processed = data_processed[data_processed[text_col].parallel_apply(is_english)]
    non_english_reviews_count = non_english_reviews_count - data_processed.shape[0]
    logger.info(f"Number of non-English observations: {non_english_reviews_count}")

    # Tokenize and preprocess the text
    logger.info("Processing text - lemmatization, stop-words removal, punctuation removal and others")
    data_processed["tokens"] = data_processed[text_col].parallel_apply(preprocess_text)

    return data_processed
