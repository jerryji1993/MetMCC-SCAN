# general modules
import os
import argparse
from copy import deepcopy
import pickle as pkl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# src
import config
from preprocessors import *
from utils import *


def get_args():
    """
    Define parser and parse input command line arguments.
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="<Required> Path to input data csv. Input table should contain feature columns specified in config.py.",
    )

    parser.add_argument(
        "--target_name",
        default='SLNB',
        type=str,
        required=True,
        help="<Required> Column name of the response variable/label.",
    )
    
    parser.add_argument(
        "--idx_num_cols",
        nargs="*",
        required=True,
        type=int,
        help="<Required> Indices of numeric columns.",
    )
    
    parser.add_argument(
        "--idx_cat_cols",
        nargs="*",
        required=True,
        type=int,
        help="<Required> Indices of categorical columns.",
    )

    parser.add_argument(
        "--dropna_thres",
        default=6,
        type=int,
        help="Threshold for number of NAs in a row to be dropped.",
    )
    
    parser.add_argument(
        "--decision_thres",
        default=0.1,
        type=float,
        help="Threshold for classifying positive vs. negative classes.",
    )
    
    parser.add_argument(
        "--test_frac",
        default=0.1,
        type=float,
        help="Fraction of test set during train_test split"
    )
    
    parser.add_argument(
        "--beta",
        default=0.2,
        type=float,
        help="Beta value used to compute negative F-beta score. beta < 1 favors more on NPV while beta > 1 more on specificity"
    )
    
    parser.add_argument(
        "--n_iter_bayesopt",
        default=20,
        type=int,
        help="Number of iterations for bayesian optimization"
    )
    
    parser.add_argument(
        "--random_seed",
        default=None,
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        default=0,
        type=int,
        help="Verbosity level"
    )
    
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of threads to use"
    )

    parser.add_argument(
        "--model_dir",
        default='../model',
        type=str,
        help="Path to save model and preprocessor.",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    
    if not os.path.exists(args.model_dir):
        if args.verbose > 0:
            print("* Input path to save model does not exist. Creating...")
            os.makedirs(args.model_dir)
            
    # reading and preprocessing data
    if args.verbose > 0:
        print("* Reading data...")
    df = pd.read_csv(args.data_dir)
    if args.verbose > 1:
        print("* Selecting features...")
    df = df[config.FEATURES]
    if args.verbose > 1:
        print("* Binning categories...")
    df = binning_categories(df)
    if args.verbose > 1:
        print("* Dropping NA rows...")
    df.dropna(thresh=args.dropna_thres, inplace=True)
    
    # Get X and y
    X = np.array(df.loc[:, df.columns != args.target_name])
    y = np.array(df.loc[:, df.columns == args.target_name])
    
    if (not args.idx_num_cols)&(not args.idx_cat_cols):
        raise ValueError("Indices for numeric and categorical columns cannot all be empty")
    
    if args.verbose > 0:
        print("* Preprocessing data...")
    X = preprocess(X, args.idx_num_cols, args.idx_cat_cols, model_dir = args.model_dir)
    
    if args.verbose > 0:
            print("* Splitting data into {} training and {} testing dataset".format(1-args.test_frac,
                                                                                    args.test_frac
                                                                                   ))
    if args.random_seed is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = args.test_frac, 
                                                            random_state = args.random_seed) # set seed to ensure reproducibility
        clf = RandomForestClassifier(oob_score=True,
                                 random_state=args.random_seed,
                                 class_weight = 'balanced')
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = args.test_frac)
        clf = RandomForestClassifier(oob_score=True,
                         class_weight = 'balanced')
    

    param_grid = {
                    'n_estimators': (100,2000),
                    'max_depth': (3, 50),  
                    'criterion': ['gini','entropy'],
                    'max_features': (1,X_train.shape[1]-1),
                   }
    
    if args.verbose > 0:
            print("* Bayesian optimization for hyperparam tuning")
    opt_rf = bayesian_optimize(clf, param_grid,
                              X_train, X_test, 
                              y_train, y_test,
                              beta = args.beta,
                              threshold = args.decision_thres,
                              n_iter = args.n_iter_bayesopt,
                              verbose = args.verbose, 
                              n_jobs = args.n_threads)
    
    
    # pickling model
    if args.verbose > 0:
            print("* Saving tuned model to: {}".format(args.model_dir+'/rf_model.pkl'))
    with open(args.model_dir+'/rf_model.pkl', 'wb') as f:
        pkl.dump(opt_rf, f)
    
if __name__ == "__main__":
    main()

