import pandas as pd
from datetime import datetime

def load_data(path):
    df = pd.read_csv(path)
    return df

def add_new_columns(df):
