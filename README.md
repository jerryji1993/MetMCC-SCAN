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
## Environment setup
To install all requirements for running `src/train_model.py`, simply run:
```
python3 -m pip install -r src/requirements.txt
```
Also for deployment:
```
python3 -m pip install -r heroku/requirements.txt
```
Creating a new `conda` environment or virtual env is highly recommended.

## train model
The model needs to be trained before deploying to Heroku server. To train the model from scratch, run:
```
cd src
python train_model.py \
  --data_dir ../data/YOUR_DATA_HERE.csv \
  --target_name SLNB \
  --idx_num_cols 0 3 4 \
  --idx_cat_cols 1 2 5 6 7 8 9 \
  --random_seed 123 \
  --verbose 2 \
  --n_threads 5 \
  --n_iter_bayesopt 1

```
For the full list of input args, check out `src/train_model.py`. Note that the dataset I used is protected so I could not upload it here, however please make sure that the dataset you use contains the same columns as those in `src/config.py`. Please also modify `--idx_num_cols` and `--idx_cat_cols` accordingly based on your data.

Once the model is trained (via Bayesian Optimization for hyperparameter tuning), it will be saved to `../model` by default. The resulting serialized model and preprocessor are then ready to be used for deployment.

## Running streamlit app on your local machine
To run Streamlit app on your local machine, simply run:
```
cd ../heroku
streamlit run app.py
```

## Deploying streamlit app to Heroku server
To deploy streamlit app to Heroku server, you will first need to create a `setup.sh` file and put it in the `heroku/` folder. The `setup.sh` file has the following format:
```
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
```
where you need to change the email address to the correct email. Then, clone the repository, download `Heroku CLI` and follow the standard steps to push to Heroku server.

## Presentation and demo link
Presentation slides to this work can be found [here](https://docs.google.com/presentation/d/1ar9YK1E1geHDJsOuPARh0FYWF4-pmbSlVwKKdWI2ssk/edit?usp=sharing). An example demo server I created can be found [here](http://biopsy.digital/).

## Author
Yanrong (Jerry) Ji

Insight Health Data Fellow, Boston 20C

jerry@mygeno.me
