########################################################################
# Wrapper functions and functions for building the topic modelling model
########################################################################


from typing import List, Tuple, Dict, Iterable, Union, Any

import gensim
import pandas as pd
from gensim import corpora, models


def prepare_corpus(
    doc_clean: Iterable[str],
    min_doc_frequency: int,
    max_doc_frequency: float,
    max_terms: int,
) -> tuple[corpora.Dictionary, List[Union[Tuple[List[Tuple[int, int]], Dict[Any, int]], List[Tuple[int, int]]]]]:

    """
    Create a term dictionary for the cleaned document tokens and convert them into a Document Term Matrix.

    Args:
        doc_clean (list of lists): List of tokenized and cleaned documents.
        min_doc_frequency (int): Minimum document frequency for terms to be included.
        max_doc_frequency (float): Maximum document frequency for terms to be included.
        max_terms (int): Maximum number of terms to keep.

    Returns:
        dictionary (corpora.Dictionary): The term dictionary for the corpus.
        corpus (list of lists of tuples): The Document Term Matrix.
    """

    # Create the term dictionary for the corpus, where each unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Filter extreme terms based on document frequency thresholds.
    dictionary.filter_extremes(
        no_below=min_doc_frequency,
        no_above=max_doc_frequency,
        keep_n=max_terms,
    )

    # Convert the list of documents (corpus) into a Document Term Matrix using the prepared dictionary.
    corpus = [dictionary.doc2bow(tokens) for tokens in doc_clean]

    return dictionary, corpus


def grid_search_lsi(
        corpus_tfidf: List[Union[Tuple[List[Tuple[int, int]], Dict[Any, int]], List[Tuple[int, int]]]],
        dictionary: corpora.Dictionary,
        num_topics_list: List[int],
        chunksize_list: List[int],
) -> Dict[str, models.LsiModel]:

    """
    Perform grid search for LSI models with different hyperparameters.

    Args:
        corpus_tfidf (list of lists of tuples): TF-IDF transformed corpus.
        dictionary (corpora.Dictionary): The term dictionary for the corpus.
        num_topics_list (list of int): List of candidate numbers of topics.
        chunksize_list (list of int): List of candidate chunk sizes.

    Returns:
        model_dict (dict): A dictionary mapping model names to LSI models.
    """

    # Container for models
    model_dict = {}

    for num_topics in num_topics_list:
        for chunksize in chunksize_list:
            model_name = f'lsi_model_{num_topics}_{chunksize}'
            print(f"Estimating model: {model_name}...")
            lsi_model = models.LsiModel(
                corpus=corpus_tfidf,
                id2word=dictionary,
                num_topics=num_topics,
                chunksize=chunksize,
                onepass=False,
            )
            model_dict[model_name] = lsi_model

    return model_dict


def get_coherence_models(
        models_grid_search: dict[str, models.LsiModel],
        tokens: Iterable[str],
        dictionary: corpora.Dictionary,
):

    """
    Sort a list of topic modelling models by their Coherence scores in descending order.

    Args:
        models_grid_search (dictionary of gensim.models.LsiModel): Dictionary of models.
        tokens (list of lists of str): List of tokenized documents.
        dictionary (corpora.Dictionary): The term dictionary for the corpus.

    Returns:
        sorted_models (list of tuple): List of (model_name, coherence_score) tuples, sorted by coherence score.
    """

    # Initialize an empty dictionary to store Coherence scores for each model
    coherence_scores = {}

    # Loop through the models and calculate their Coherence scores
    for model_name in models_grid_search:

        coherence_model = models.CoherenceModel(
            model=models_grid_search[model_name],
            texts=tokens,
            dictionary=dictionary,
            coherence='c_v',
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores[model_name] = coherence_score

    # Sort the models by Coherence score in descending order
    sorted_models = sorted(coherence_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the coherence score for each model
    for model_name, coherence_score in sorted_models:
        print(f'Model: {model_name}, Coherence Score: {coherence_score}')

    return sorted_models


def prepare_output(
        df: pd.DataFrame,
        dictionary: corpora.Dictionary,
        model_estimated: Union[models.ldamulticore.LdaMulticore, models.LsiModel, models.ldamodel.LdaModel],
):

    """
    Assign topics to text data based on a given LSI or LDA model.

    Args:
        df (pandas.DataFrame): DataFrame containing the text data.
        dictionary (corpora.Dictionary): The term dictionary for the corpus.
        model_estimated: An LSI or LDA model trained on the document-term matrix.

    Returns:
        df (pandas.DataFrame): DataFrame with assigned topics for each text.
    """

    df = df.copy(deep=True)

    def classify_title(
            tokens
    ):
        """
        Classify a list of tokens into a topic using the provided LSI or LDA model.

        Args:
            tokens (list of str): List of tokenized and cleaned text.

        Returns:
            topic_id (int): The topic ID assigned to the text.
        """
        topic_scores = model_estimated[dictionary.doc2bow(tokens)]
        try:
            sorted_topics = sorted(topic_scores, key=lambda x: -x[1])  # Sort by topic score in descending order
            top_topic = sorted_topics[0][0]  # Get the topic with the highest score
        except IndexError:
            top_topic = ""
        return top_topic

    df['NLP_output'] = df['tokens'].apply(classify_title)
    topic_keywords = {}
    topics = model_estimated.print_topics(num_words=3)
    for topic_id, topic in topics:
        topic_keywords[topic_id] = topic

    df['NLP_output'] = df['NLP_output'].map(topic_keywords)
    return df
