import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads data
    Steps followed are:
    1. Load messages dataset
    2. Load categories dataset
    3. Merge message and categories dataset

    Parameters 
    ----------
    messages_filepath : str
        file name for messare csv file
    categories_filepath : str
        filename for categories csv

    Returns
    -------
    DataFrame
        Merged messages and categories    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """
    Cleans data. 
    Main cleanups done are:
    1. create a dataframe of the 36 individual category columns
    2. Set a numeric value for each category based on the category string
    3. update original df with the new category columns

    Parameters 
    ----------
    df: DataFrame
        Dataset to be cleaned
    Returns
    -------
    DataFrame
        Cleaned dataframe    
    """ 
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe and 
    # extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # drop duplicates
    df.drop_duplicates(inplace=True);
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat ([df,categories], axis=1)
    
    return df


def save_data(df, database_filename):
    """
    Saves data in database
    
    Parameters 
    ----------
    df : DataFrame
        Dataset to be saved in db  
    database_filename: str
        database filename
    """
    conn_str = 'sqlite:///' + database_filename
    engine = create_engine(conn_str)
    engine.execute("DROP TABLE IF EXISTS msg_categories_tbl")
    df.to_sql('msg_categories_tbl', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()