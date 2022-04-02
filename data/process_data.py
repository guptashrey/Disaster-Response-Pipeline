import sys
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges 2 datasets.
    
    Parameters:
    messages_filepath: messages csv file
    categories_filepath: categories csv file
    
    Returns:
    df: merged dataframe of messages and categories datasets
    
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='outer', on ='id')
    return df

def clean_data(df):
    """
    Cleans the dataframe as follows:
    1. Splits categories into separate columns
    2. Convert category values to 0 or 1
    3. Removes duplicates
    
    Parameters:
    df: DataFrame
    
    Returns:
    df: Cleaned DataFrame
    """

    categories = df['categories'].str.split(";", expand=True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    categories['related'][categories['related']==2]=1
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    Stores df in a SQLite database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  

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