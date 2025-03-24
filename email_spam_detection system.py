# pandas library for dataset manipulation
"""
Purpose: Data Handling & Analysis
Usage in Spam Detection:

Reads and processes email datasets (usually in CSV format).

Organizes email text and labels (spam vs. non-spam) into structured tables.

Enables efficient data manipulation (e.g., filtering, grouping, cleaning).
"""
import os.path

import pandas as pd

# numpy library for numerical computation
"""
Purpose: Numerical Computation
Usage in Spam Detection:

Efficiently handles large arrays and matrices (useful for storing feature vectors).

Often used in machine learning models for computations.

Helps in handling vectorized operations when transforming text into numerical representations (e.g., TF-IDF, word embeddings).
"""
import numpy as np

# nlkt(Natural language toolkit)
"""
Purpose: Removes commonly used words (stopwords) like "the," "and," "is," which do not contribute to spam detection.

Usage in Spam Detection:

Improves the accuracy of models by reducing noise in text data.

Helps focus on important words that indicate spam (e.g., "win," "free," "click").
"""
from nltk.corpus import  stopwords

# nlkt(Natural language toolkit)
"""
Purpose: Splits email text into individual words (tokens).

Usage in Spam Detection:

Helps in feature extraction by breaking down an email into meaningful words.

Essential for applying further processing like stemming, lemmatization, or TF-IDF vectorization.
"""
from nltk import word_tokenize

# string library
"""
Purpose: Handles text cleaning by removing punctuation.
Usage in Spam Detection:

Spam emails may contain extra symbols (e.g., "!!!FREE!!!") to evade filters.

Removing punctuation helps in standardizing the text for analysis.
"""
import  string


#load dataset using panda stored in the ./data directory
dataframe_emails = pd.read_csv("data/emails.csv")

# overview of dataset
print(dataframe_emails.columns)
print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(dataframe_emails.head())
