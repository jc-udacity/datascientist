import pandas as pd
import numpy as np

df = pd.read_csv('./survey_results_public.csv')
schema = pd.read_csv('./survey_results_schema.csv')


## A Look at the Data
# Solution to Question 1
num_rows = df.shape[0]
num_cols = df.shape[1]

# Solution to Question 2
no_nulls = set(df.columns[df.isnull().mean()==0])

# Solution to Question 3
most_missing_cols = set(df.columns[df.isnull().mean() > 0.75])
