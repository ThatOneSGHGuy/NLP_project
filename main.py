# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:00:13 2023

@author: mikolaj
"""

import argparse
import pickle

# import nltk
import yaml
from gensim import models

from utilities.eda import *
from utilities.preprocessing import preprocess_texts_eng
from utilities.topic_modelling import (
    grid_coherence_models,
    grid_search_lsi,
    prepare_corpus,
    prepare_output,
)

config = yaml.safe_load(open("config.yaml", encoding="utf-8"))

PATH = Path(config.get("PATH"))
DATA_PATH = PATH / "data"
IMG_PATH = PATH / "img"
NLTK_CACHE = ".nltk_cache"
MODEL_PATH = PATH / "model"
IN_FILE = "df_raw.csv"
PROCESSED_FILE = "df_processed.csv"
OUT_FILE = "df_output.csv"

# make sure the directories exist
(PATH / NLTK_CACHE).mkdir(exist_ok=True, parents=True)
(PATH / IMG_PATH).mkdir(exist_ok=True, parents=True)
(PATH / MODEL_PATH).mkdir(exist_ok=True, parents=True)

# Download NLTK resources (if not already downloaded)
# nltk.download("punkt", download_dir=PATH / NLTK_CACHE)
# nltk.download("stopwords", download_dir=PATH / NLTK_CACHE)
# nltk.download("wordnet", download_dir=PATH / NLTK_CACHE)
# nltk.download("omw-1.4", download_dir=PATH / NLTK_CACHE)
# nltk.download("averaged_perceptron_tagger", download_dir=PATH / NLTK_CACHE)

parser = argparse.ArgumentParser(description="Running topic modelling pipeline")
parser.add_argument("--eda", action="store_true", help="Run EDA")
parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
parser.add_argument("--model", action="store_true", help="Run model estimation")
# console: python main.py --eda --preprocess --model

if __name__ == "__main__":
    steps = parser.parse_args()

    if steps.eda:
        ##############
        # Part 1 - EDA
        ##############

        logger.info("Running EDA")
        # Check what delimiter is used in the file - open the file with a Notebook to check
        data_raw = pd.read_csv(DATA_PATH / IN_FILE, sep=",")
        logger.info(f"Data loaded from {str(DATA_PATH / IN_FILE)}.")
        logger.info(f"Data shape: {data_raw.shape}")

        # Check the count of unique values per column
        logger.info(f"{'=' * 50}")
        logger.info(f"Unique count per columns:")
        for col in data_raw.columns:
            logger.info(f"{col:<25}: {data_raw[col].nunique()}")

        # Check the unique values for columns with low number of unique values and print their frequencies
        logger.info(f"{'=' * 50}")
        logger.info(f"Count per columns for low frequency unique:")
        low_count_describe(10, data_raw)
        logger.info(f"{'=' * 50}")

        # Check for the duplicates
        logger.info("Quality checks")
        dups = data_raw.duplicated().sum()
        if dups == 0:
            logger.info(f"Duplicates: No duplicates found.")
        else:
            logger.warning(f"Duplicates: {dups} duplicates found in raw data")

        # Check for NA(s)
        check_na(df=data_raw)
        # Check for zeros (possible label for NA(s))
        check_zeros(df=data_raw)
        # Check if specified columns are of numberic dtype
        check_numeric_col(df=data_raw, expected_numeric_cols=["ReviewStars", "ReviewForPromo", "id"])

        # Transform dates into Python datetime format
        data_raw = date_transform(
            df=data_raw,
            date_col="ReviewDate",
            date_format="%Y-%m-%d",
        )

        # Plot to investigate data distribution by date
        plot_review_count_by_date(df=data_raw, img_path=IMG_PATH)
        # Plot to investigate data distribution by the review length
        plot_distribution_review_lengths(df=data_raw, img_path=IMG_PATH)

    if True:
        #############################
        # Part 2 - Text preprocessing
        #############################

        data_raw = pd.read_csv(DATA_PATH / IN_FILE, sep=",")
        logger.info(f"Data loaded from {str(DATA_PATH / IN_FILE)}.")

        # Run the wrapper function to preprocess the text in the ReviewText column
        data_processed = preprocess_texts_eng(
            df=data_raw,
            text_col="ReviewText",
            shuffle=True,
            train_subset=1,
            workers=4,
        )
        # Create a dictionary and a corpus
        dictionary, corpus = prepare_corpus(
            doc_clean=data_processed["tokens"],
            min_doc_frequency=10,
            max_doc_frequency=0.3,
            max_terms=None,
        )
        data_processed.to_csv(DATA_PATH / PROCESSED_FILE, sep=",", index=False)
        logger.info(f"Processed data saved to {str(DATA_PATH / PROCESSED_FILE)}")

        with open(MODEL_PATH / "dictionary.pkl", "wb") as f:
            pickle.dump(dictionary, f)
        logger.info(f"Dictionary saved to {str(MODEL_PATH / 'dictionary.pkl')}")

        with open(MODEL_PATH / "corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        logger.info(f"Corpus saved to {str(MODEL_PATH / 'corpus.pkl')}")

    if True:
        ########################################
        # Part 3 - Model building and estimation
        ########################################

        data_raw = pd.read_csv(DATA_PATH / IN_FILE, sep=",")
        logger.info(f"Raw data loaded from {str(DATA_PATH / IN_FILE)}.")

        data_processed = pd.read_csv(DATA_PATH / PROCESSED_FILE, sep=",")
        logger.info(f"Processed data loaded from {str(DATA_PATH / PROCESSED_FILE)}.")

        with open(MODEL_PATH / "dictionary.pkl", "rb") as f:
            dictionary = pickle.load(f)
        logger.info(f"Dictionary loaded from {str(MODEL_PATH / 'dictionary.pkl')}")

        with open(MODEL_PATH / "corpus.pkl", "rb") as f:
            corpus = pickle.load(f)
        logger.info(f"Corpus loaded from {str(MODEL_PATH / 'corpus.pkl')}")

        # Build the TF-IDF model
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        # Define the hyperparameter combinations to search
        num_topics_list = [10, 20, 50]
        chunksize_list = [25, 50, 100]

        # Perform grid search for LSI models
        models_grid_search_lsi = grid_search_lsi(
            corpus_tfidf=corpus_tfidf,
            dictionary=dictionary,
            num_topics_list=num_topics_list,
            chunksize_list=chunksize_list,
            workers=6,
        )

        # Evaluate LSI models using Coherence score
        sorted_models = grid_coherence_models(
            models_grid_search=models_grid_search_lsi,
            tokens=data_processed["tokens"],
            dictionary=dictionary,
        )

        # Create an alternative LDA model to compare the LSI models to
        lda_model = models.LdaMulticore(
            corpus=corpus_tfidf,
            id2word=dictionary,
            passes=20,
            num_topics=15,
            workers=8,
        )

        # Get the coherence score for the LDA model
        coherence_model_LDA = models.CoherenceModel(
            model=lda_model,
            texts=data_processed["tokens"],
            dictionary=dictionary,
            coherence="c_v",
        )

        coherence_score_LDA = coherence_model_LDA.get_coherence()
        print(coherence_score_LDA)

        ########################################
        # Part 4 - Final model choice and output
        ########################################

        final_model = models_grid_search_lsi["lsi_model_10_100"]

        # Create a final df to store the assigned topics
        data_final = prepare_output(data_processed, dictionary, final_model)
        data_final = data_final.merge(
            data_raw,
            how="right",
            on=[
                "ItemNumber",
                "Retailer",
                "ReviewTitle",
                "ReviewText",
                "ReviewStars",
                "ReviewForPromo",
                "Category",
                "SubCategory",
                "BrandName",
                "id",
            ],
        )
        data_final.drop(columns="ReviewDate_y", inplace=True)
        data_final.rename(columns={"ReviewDate_x": "ReviewDate"}, inplace=True)

        # Output the final df to .csv file
        data_final.to_csv(PATH / OUT_FILE, sep=",", index=False)
