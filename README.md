# Disaster Response Pipeline Project
## Project Description
The objective of this project is yo build a model for a webservice to classify disaster messages.   
The three main parts of the project are:
1. Data processing (ETL Pipeline): 
    - For data processing and cleansing we used pandas and related libraries.
2. Text processing and classification (ML Pipeline):
    - We use NLTK for text processing and sklearn for classification.
3. Create a classifier webservice:
    - To expose the classification servise as a web services we used flask framework.

## Files in the repository
### File tree
```
├── app                           # Web services folder   
│   ├── run.py                    # Flask web application   
│   └── templates                 # Templates folder     
│       ├── go.html               # Template file   
│       └── master.html           # Container template file   
├── data                          # ETL pipeline folder    
│   ├── disaster_categories.csv   # Data file for categories   
│   ├── disaster_messages.csv     # Data file for messages   
│   ├── DisasterResponse.db       # Local Only (database file)    
│   └── process_data.py           # ETL pipeline implementation   
├── models                        # ML Pipeline folder   
│   ├── classifier.pkl            # Local only (saved trained model)   
│   └── train_classifier.py       # ML pipeline implementation   
└── README.md      
```
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/ data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
