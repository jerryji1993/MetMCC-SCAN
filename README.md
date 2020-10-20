# MetMCC-SCAN
This repository contains information and implementation of MetMCC-SCAN, which is a computational predictive model of metastasis of Merkel Cell Carcinoma. This was finished as an Insight 20C Health Data Science project consulting for Oregon Health and Science University in 3 weeks.

## Problem statement
Merkel Cell Carcinoma is a rare and deadly form of skin cancer that is characterized by fast growth and spread to other parts of body. Currently, metastasis of Merkel Cell Carcinoma is confirmed by Sentinel Lymph Node Biopsy (SLNB), which is 100% invasive, has 11% complication rate and costs $15,000 per patient. Nevertheless, more than 80% of patients will obtain a negative result, for whom the biopsy is somewhat inefficient and unnecessary considering the associated costs and health risks.

## The solution
Working as a consultant for a research hospital, I used clinical and demographic data from cancer patients in National Cancer Database to develop a Random Forest model to computationally automate the prediction of metastasis of Merkel Cell Carcinoma, so that patients who do not need an actual biopsy can be pre-filtered. The model was deployed as a Streamlit web application on Heroku for the clinicians to interactively use. Compared to the current standard practice in clinic as baseline, my model enabled potential reduction of unnecessary biopsies by 37.4%.

## Repository structure
```
.
├── README.md                           <- You are here
│
├── src                                 <- source codes to train the model
│   ├── config.py                       <- configurations
│   │  
│   ├── preprocessors.py                <- custom-defined preprocessing class and functions
│   │  
│   ├── requirements.txt                <- Python package dependencies to run src/train_model.py
│   │
│   ├── train_model.py                  <- main training script with command-line args
│   │
│   ├── utils.py                        <- util functions used in train_model.py
│
├── ipynb_notebooks                     <- contains jupyter notebooks for EDA, interpretation, etc.
│   ├── EDA.ipynb                       <- EDA notebook
│   │  
│   ├── feature_selection.R             <- R script for feature selection
│   │  
│   ├── interpret.ipynb                 <- notebook for model interpretation
│   │  
│   ├── model_fitting.ipynb             <- deprecated model fitting notebook
│   
├── heroku                              <- contains scripts for building web app and deployment on heroku
│   ├── image                           <- contains image that for web app frontend
│   │  
│   ├── model                           <- models
│   │   ├── preprocessor_sepImpute.pkl  <- previously trained preprocessor
│   │   │ 
│   │   ├── rf_bias_..._binned.pkl      <- previously trained Random Forest model
│   │
│   ├── src                             <- modules to be imported to app
│   │   ├── preprocessors.py            <- custom-defined preprocessing class and functions
│   │
│   ├── Procfile                        <- command to start streamlit app on Heroku server
│   │
│   ├── app.py                          <- main script for the streamlit app
│   │
│   ├── config.py                       <- configurations
│   │
│   ├── load_css.py                     <- function to loading css in streamlit
│   │
│   ├── requirements.txt                <- Python package dependencies to run heroku/app.py
│   │
│   ├── style.css                       <- css file for style of banner
│ 
```
