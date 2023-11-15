# NLP_project
An end-to-end solution to NLP task (topic modeling) - starting from EDA, preprocessing to estimation of a number of seperate models and evaluating them. Due to the nature of this project, data could not be published.

# Steps

There are 3 steps in this project:

1. EDA - In the EDA step, several important features of the data are being checked. First of all, I am looking at different types of count data. Additionally, quality checks are introduced by checking for zeros/nulls and validating column types. At the end a couple of plots are shown to investigate some of the data distributions.

2. Preprocessing - The first part of preprocessing focuses on removing short texts, optionally shuffling the data, filtering by language, and tokenizing the text. Second part prepares a corpus from the already-cleaned text and exports the corpus (and a dictionary) to an external file.

3. Modelling - In this section TF-IDF corpus model is prepared and fed into LSI and LDA models. Additionally, for LSI grid search is performed. Both models are evaluated based on Coherence score. At the end, final model is chosen along with the predictions and the results are exported into an external file.
