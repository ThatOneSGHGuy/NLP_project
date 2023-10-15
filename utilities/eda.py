###################
# Functions for EDA
###################
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger
from typing import List

import matplotlib.pyplot as plt


def low_count_describe(
        low_count_threshold: int,
        df: pd.DataFrame,
) -> None:
    """
    Performs exploratory data analysis (EDA) on columns with low unique value counts in a DataFrame.

    Args:
        low_count_threshold (int): The maximum number of unique values a column can have to be considered low count.
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        None: Prints the unique value counts for columns with low unique value counts along with the unique values.
    """
    df = df.copy(deep=True)
    low_cols = df.nunique()[df.nunique() <= low_count_threshold].index.values

    # print hyphens unitl 80 characters
    logger.info(f"{'-' * 50}")
    for col in low_cols:
        logger.info(f"Column: {col}")
        cnts = df[col].value_counts()
        for val in cnts.index:
            logger.info(f"\t{val:<25}: {cnts[val]}")
        logger.info(f"{'-' * 50}")


def date_transform(
        df: pd.DataFrame,
        date_col: str,
        date_format: str,
) -> pd.DataFrame:
    """
    Transforms a date column in a DataFrame to a datetime format.

    Args:
        df (pd.DataFrame): The DataFrame containing the date column.
        date_col (str): The name of the date column to transform.
        date_format (str): The format of the date values in the column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with the specified date column transformed to datetime format.
    """

    logger.info(f"Converting `{date_col}` to dateime format.")
    df = df.copy(deep=True)
    df[date_col] = pd.to_datetime(
        df[date_col],
        format=date_format,
        errors='coerce'
    )

    return df


def check_na(
        df: pd.DataFrame,
) -> None:
    """
    Checks for NA values in the DataFrame for each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        None: Prints the number of found NA values, if there are any in a given column.
    """

    df = df.copy(deep=True)

    nans = df.isna().sum()
    logger.info(f"{'-' * 50}")
    if nans.sum() == 0:
        logger.info(f"NAs: No NA(s) found.")
    else:
        for col, n_nans in nans.items():
            if n_nans > 0:
                logger.warning(f"NAs: {col} has {n_nans} NA(s).")
                logger.info(f"{'-' * 50}")


def check_zeros(
        df: pd.DataFrame,
) -> None:
    """
    Checks for 0s in the DataFrame for each column.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        None: Prints the number of found 0s, if there are any in a given column.
    """

    df = df.copy(deep=True)

    zeros = df.isin([0]).sum(axis=0)
    logger.info(f"{'-' * 50}")
    if zeros.sum() == 0:
        logger.info(f"Zeros: No zeros found.")
    else:
        for col, n_zeros in zeros.items():
            if n_zeros > 0:
                logger.warning(f"Zeros: {col} has {n_zeros} zero(s).")
                logger.info(f"{'-' * 50}")


def check_numeric_col(
        df: pd.DataFrame,
        expected_numeric_cols: List[str],
) -> None:
    """
    Checks if a specified list of columns in a dataframe contains only numeric values.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        expected_numeric_cols (list of strings): Column names for which the values are expected to be only numeric.

    Returns:
        None: Prints whether specified column contains numeric values or not.
    """

    df = df.copy(deep=True)

    logger.info(f"{'-' * 50}")
    if expected_numeric_cols:
        for col in expected_numeric_cols:
            f_convert = lambda s: pd.to_numeric(s, errors="coerce")
            casted_srs = df[col].apply(f_convert)
            if casted_srs.notnull().all():
                logger.info(f"Expected dtype: numeric - `{col}` contains only numeric values.")
            else:
                logger.warning(f"Expected dtype: numeric - `{col}` contains non-numeric values.")
            logger.info(f"{'-' * 50}")
    else:
        logger.warning(f"Expected dtype: No numeric columns were supplied in the function call.")


def plot_review_count_by_date(
        df: pd.DataFrame,
        img_path: Path,
) -> None:
    """
    Plot number of reviews written on a given date.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        img_path (Path): Path to output images to.

    Returns:
        None: Plot is saved in a seperate file.
    """

    df = df.copy(deep=True)
    # Group the data by date and count the number of reviews for each date
    date_counts = df["ReviewDate"].value_counts().reset_index()
    date_counts.columns = ["ReviewDate", "ReviewCount"]
    date_counts = date_counts.reset_index(drop=True)

    # Sort the data by date in ascending order
    date_counts = date_counts.sort_values(by="ReviewDate")

    # Plot the number of reviews on a given date
    rev_date = date_counts["ReviewDate"]
    rev_count = date_counts["ReviewCount"]
    plt.figure(figsize=(18, 9))
    plt.bar(rev_date, rev_count, color="skyblue")
    plt.title("Number of Reviews by Date")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(img_path / "review_count_by_date.png")
    logger.info(f"Chart 'review_count_by_date.png' saved to folder {str(img_path)}\\")

    # Get the number of reviews with a date before 2020 - more than 93% of the reviews were written before 2020,
    # although the first record is from 2009.
    logger.info(f"First review date: {rev_date.min()}")
    logger.info(f"Last review date: {rev_date.max()}")

    # Filter the DataFrame to select reviews with a date before 2020
    reviews_before_2020 = (rev_date < "2020-01-01").sum()
    logger.info(f"Share of the reviews before 2020 in the total number of reviews:"
                f" {reviews_before_2020 / df.shape[0]:.2%}"
                )


def plot_distribution_review_lengths(
        df: pd.DataFrame,
        img_path: Path,
) -> None:
    """
    Plot the distribution of the lengths of reviews.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        img_path (Path): Path to output images to.

    Returns:
        None: Plot is saved in a seperate file.
    """

    df = df.copy(deep=True)
    # Calculate the number of words in each review and store it in 'count' variable
    len_review = df["ReviewText"].str.split().str.len()
    count_data = Counter(len_review)

    # Create a histogram with specified range and bins
    plt.figure(figsize=(18, 9))
    plt.bar(
        count_data.keys(),
        count_data.values(),
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Number of Words in Review")
    plt.ylabel("Frequency")
    plt.title("Distribution of Review Lengths")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(range(0, 101, 5))  # Set x-axis ticks to display every 5 units
    plt.yticks(range(0, 3501, 250))  # Set x-axis ticks to display every 250 units
    plt.xlim(0, 100)  # Set x-axis limit from 0 to 100
    plt.savefig(img_path / "distribution_of_reviews_length.png")
    logger.info(f"Chart 'distribution_of_reviews_length.png' saved to folder {str(img_path)}\\")
