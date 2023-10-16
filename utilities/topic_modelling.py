########################################################################
# Wrapper functions and functions for building the topic modelling model
########################################################################

from functools import partial
from typing import List, Tuple, Dict, Iterable, Union, Any
import multiprocessing

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
        workers: int = 1,
        **kwargs,
) -> List[Dict[str, models.LsiModel]]:
    """
    Perform grid search for LSI models with different hyperparameters.

    Args:
        corpus_tfidf (list of lists of tuples): TF-IDF transformed corpus.
        dictionary (corpora.Dictionary): The term dictionary for the corpus.
        num_topics_list (list of int): List of candidate numbers of topics.
        chunksize_list (list of int): List of candidate chunk sizes.
        workers (int): Number of CPUs to use in the grid search.

    Returns:
        model_dict (dict): A dictionary mapping model names to LSI models.
    """

    # Container for models
    model_dict = {}
    param_grid = [({'chunksize': chunksize}, {'num_topics': num_topics})
                  for chunksize in chunksize_list
                  for num_topics in num_topics_list]
    fixed_params = dict(
        corpus=corpus_tfidf,
        id2word=dictionary,
        **kwargs,
    )

    if workers == 1:
        for params in param_grid:
            model_dict.update(
                get_model_lsi(params, **fixed_params),
            )
    else:
        # Create a pool of processes for parallel execution
        with multiprocessing.Pool(processes=workers) as Pool:
            results = list(Pool.map(
                partial(get_model_lsi, **fixed_params),
                param_grid,
            ))

        model_dict = {key: value for d in results for key, value in d.items()}

    return model_dict


def get_model_lsi(
        params,
        **kwargs,
) -> Dict:
    """
    Create and return Latent Semantic Indexing (LSI) model.

    Args:
        params (list of dicts): Dicionary of parameters containing number of topics and chunksize
            which are subject to tuning.

    Returns:
        Dict: A dictionary containing the LSI models with their associated names as keys.
    """

    params = {key: value for d in params for key, value in d.items()}
    model_dict = {}

    num_topics = params.get('num_topics')
    chunksize = params.get('chunksize')

    model_name = f'lsi_model_{num_topics}_{chunksize}'
    print(f"Estimating model: {model_name}...")

    lsi_model = models.LsiModel(
        num_topics=num_topics,
        chunksize=chunksize,
        onepass=False,
        **kwargs,
    )

    model_dict[model_name] = lsi_model

    return model_dict


def grid_coherence_models(
        models_grid_search: Dict[str, models.LsiModel],
        tokens: Iterable[str],
        dictionary: corpora.Dictionary,
        **kwargs,
) -> Dict[str, float]:
    """
    Calculate Coherence scores for each topic modelling model and sort them in descending order wrt. Coherence score.

    Args:
        models_grid_search (dictionary of gensim.models.LsiModel): Dictionary of models.
        tokens (list of lists of str): List of tokenized documents.
        dictionary (corpora.Dictionary): The term dictionary for the corpus.

    Returns:
        sorted_models (list of tuple): List of (model_name, coherence_score) tuples, sorted by coherence score.
    """

    # Initialize an empty dictionary to store Coherence scores for each model
    coherence_scores = {}

    fixed_params = dict(
        texts=tokens,
        dictionary=dictionary,
        coherence='c_v',
        **kwargs,
    )
    # Loop through the models and calculate their Coherence scores
    for model in models_grid_search.items():
        coherence_scores.update(
            get_coherence_model(model_dict=model, **fixed_params),
        )

    # Sort the models by Coherence score in descending order
    sorted_models = {k: v for k, v in sorted(coherence_scores.items(), key=lambda item: item[1], reverse=True)}

    # Print the coherence score for each model
    for model_name, coherence_score in sorted_models.items():
        print(f'Model: {model_name}, Coherence Score: {coherence_score}')

    return sorted_models


def get_coherence_model(
        model_dict: Tuple[str, models.LsiModel],
        **kwargs
) -> Dict[str, float]:
    """
    Calculate Coherence score for a given LSI model.

    Args:
        model_dict (tuple of dicts): Tuple containing LSI model name and the model itself.

    Returns:
        Dict: A dictionary containing Coherence scores with the associated model names as keys.
    """

    # Unpack the tuple
    model_name = model_dict[0]
    model = model_dict[1]

    # Create Coherence model
    coherence_model = models.CoherenceModel(
        model=model,
        **kwargs,
    )

    # Get the Coherence score
    coherence_score = {model_name: coherence_model.get_coherence()}

    return coherence_score


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
            tokens: Iterable[str],
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
