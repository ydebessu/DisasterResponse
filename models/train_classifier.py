import sys
## import libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    """
    Loads data
    Steps followed are:
    1. Read data from database
    2. Separate data into message and categories
    3. Get category names (column names)

    Parameters 
    ----------
    database_filepath : str
        file name of the database

    Returns
    -------
    DataFrame
        the list of all messages
    DataFrame
        categories
    list:
        category names    
    """
    # load data from database
    connnection_str = 'sqlite:///' + database_filepath
    tbl = 'msg_categories_tbl'  
    engine = create_engine(connnection_str)
    df = pd.read_sql(tbl, engine)
    df.drop(columns=['original', 'genre'], inplace=True)
    length = df.shape[0]
    df.dropna(inplace=True)
    print('Dropped null rows: {}'.format(length - df.shape[0]))
    X = df.message
    Y = df.drop(columns=['id', 'message'])
    
    return X, Y, list(Y.columns)


def tokenize(text):
    """
    Cleans and tokenizes a text.
    Steps followed are:
    1. Normalize text
    2. Remove panctuations
    3. Tokenize text 
    4. Remove stop words
    5. Lemmatize words and strip spaces

    Parameters 
    ----------
    text : str
        text input

    Returns
    -------
    Classifier Model : a model using pipeline or GridSearchCV 
    """
    text = re.sub(r"[^a-zA-Z0-9_]", "", text.lower())
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).strip() for w in tokens]

    return clean_tokens


def build_model():
    """
    Builds and returns a model with a given parameters

    Returns
    -------
    Classifier Model : a model using pipeline or GridSearchCV 
    """
    parameters = {
        'text_pipeline__tfidf__use_idf': [True, False]
    }
        
    return build_model_impl( parameters=parameters, use_grid_search=True)  


def build_model_impl(classifier=RandomForestClassifier(), parameters={}, use_grid_search=False):
    """
    Builds and returns a model with pipeline and grid search

    Parameters 
    ----------
    classifier : default RandomForestClassifier()
        Classifier used in MultiOutputClassifier
    parameters : dict
        Performance tunning parameters for pipeline
    use_grid_search : bool, default=False
        When set to ``True`` Grid Search is used

    Returns
    -------
    Classifier Model : a model using pipeline or GridSearchCV 
    """
    
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ('clf', MultiOutputClassifier(classifier))
    ])
    if(use_grid_search):
        model = GridSearchCV(pipeline, parameters, verbose=1)
    else:
        model = pipeline
        
    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Exports model as pickle
    
    Parameters 
    ----------
    model : Classifier
        model to be evaluated  
    X_test: DataFrame
        test data  
    Y_test: DataFrame
        actual observation categories data
    category_names: list
        names of the categories
    """
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        result = classification_report(Y_test[Y_test.columns[i]], Y_pred[:,i]) 
        print("Report for {}: \n{}".format(category_names[i], result))
        

def save_model(model, model_filepath):
    """
    Exports model as pickle
    
    Parameters 
    ----------
    model : Classifier
        model to be exported  
    model_filepath: str
        file path to save model into
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()