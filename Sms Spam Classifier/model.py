import numpy as np  
import pandas as pd 

df = pd.read_csv("spam.csv",encoding='latin-1')

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
df = df.drop_duplicates(keep='first')

import nltk
nltk.download('punkt')

df['num_chars'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))