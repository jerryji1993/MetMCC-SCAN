import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix, auc, roc_auc_score
from sklearn.metrics import roc_curve, f1_score, average_precision_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, fbeta_score, make_scorer

# Bayesian optimization
from skopt import BayesSearchCV

def binning_categories(df):
    """
    Binning the unnecessary levels/categories for categorical variables of input data
    
    Arguments:
    df -- pandas DataFrame
    
    Returns:
    df -- pandas DataFrame with binned categorical variables
    """
    # binning the categories
    df.loc[:,'SEX'] = df['SEX'].replace([1, 2],['Male', 'Female'])
    df.loc[:,'PRIMARY_SITE'] = df['PRIMARY_SITE'].replace(['C440','C441','C442','C443',
                                                           'C444','C445','C446','C447',
                                                           'C448','C449'],
                                                          ['Head_and_Neck','Head_and_Neck',
                                                           'Head_and_Neck','Other',
                                                           'Head_and_Neck','Trunk',
                                                           'Extremity','Extremity',
                                                           'Other','Other'])

    df.loc[:,'LYMPH_VASCULAR_INVASION'] = df['LYMPH_VASCULAR_INVASION'].replace([0, 1, 8, 9],
                                                                                ['No','Yes',np.nan,np.nan])

    df.loc[:,'TUMOR_INFILTRATING_LYMPHOCYTES'] = df['TUMOR_INFILTRATING_LYMPHOCYTES'].replace([0, 10, 20, 30, 999],
                                                                                              ['Negative',
                                                                                               'Weak',
                                                                                               'Strong',
                                                                                               'Present',
                                                                                               np.nan])

    df.loc[:,'IMMUNE_SUPPRESSION'] = df['IMMUNE_SUPPRESSION'].replace([0, 10, 20, 30, 40,
                                                                       50, 60, 70, 999],
                                                                      ['Negative',
                                                                       'Positive',
                                                                       'Positive',
                                                                       'Positive',
                                                                       'Positive',
                                                                       'Positive',
                                                                       'Positive',
                                                                       'Positive',
                                                                       np.nan])

    df.loc[:,'GROWTH_PATTERN'] = df['GROWTH_PATTERN'].replace([10, 20, 999],
                                                              ['Circumscribed_nodular',
                                                               'Diffusely_infiltrative',
                                                               np.nan])

    df.loc[:,'TUMOR_BASE_TRANSECTION'] = df['TUMOR_BASE_TRANSECTION'].replace([0, 10, 20, 999],
                                                                              ['Not_found',
                                                                               'Transected',
                                                                               'Not_transected',
                                                                               np.nan])

    df.loc[:,'DEPTH'] = df['DEPTH'].replace([999],np.nan)
    df.loc[:,'TUMOR_SIZE'] = df['TUMOR_SIZE'].replace([999],np.nan)
    return df



def adjusted_classes(y_scores, t):
    """
    Adjust the decision threshold to t and make binary predictions.
    
    Arguments:
    y_scores -- np.array, output probabilities from classification model.
    t -- float, decision threshold between 0 and 1.
    
    Returns:
    seq -- str, original sequence.
    """
    # play with decision threshold
    return np.where(y_scores > t, 1, 0)



def negative_pred_value(y_true, y_pred):
    """
    Computes negative predictive value.
    
    Arguments:
    y_true -- np.array, vector of true labels.
    y_pred -- np.array, vector of predicted labels.
    
    Returns:
    NPV -- number of true negative / (number of true negative + number of false negative)
    """
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    return TN/(TN+FN)



def adjusted_neg_Fbeta_score(y_true, y_prob, beta, threshold):
    """
    Computes F-beta score based on the adjusted decision threshold
        for the negative class.
    
    Arguments:
    y_true -- np.array, vector of true labels.
    y_pred -- np.array, vector of predicted labels.
    beta -- float, beta < 1 favors more on NPV, while beta > 1 more on specificity
    threshold -- float, Threshold for classifying positive vs. negative classes.
    
    Returns:
    Fb -- The computed F-beta score.
    """
    # first get adjusted hard class
    y_pred = np.where(y_prob>threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    if TN+FN == 0:
        NPV = 0
    else:
        NPV = TN/(TN+FN)
    if TN+FP == 0:
        specificity = 0
    else:
        specificity = TN/(TN+FP)
        
    if NPV+specificity == 0:
        Fb = 0
    else:
        Fb = (1+beta**2)*NPV*specificity/((beta**2)*NPV+specificity)
    return Fb



def summarize_res(model, y_prob, y_test):
    """
    Computes and prints a summary for model performance.
    """
    y_pred = adjusted_classes(y_prob[:,1], 0.1)
    print("* best parameter set: %s" % model.best_params_)
    print("* val. score: %s" % model.best_score_)
    print("* test accuracy: %s" % accuracy_score(y_test, y_pred))
    print("* test NPV: %s" % negative_pred_value(y_test, y_pred))
    print("* test Precision: %s" % precision_score(y_test, y_pred))
    print("* test Recall: %s" % recall_score(y_test, y_pred))
    print("* test AUC: %s" % roc_auc_score(y_test, y_prob[:,1]))
    print("* test F1: %s" % f1_score(y_test, y_pred))
    print("* test AUPR: %s" % average_precision_score(y_test, y_prob[:,1]))

    
    
def bayesian_optimize(clf, param_grid,
                      X_train, X_test, 
                      y_train, y_test,
                      beta = 0.2,
                      threshold = 0.1,
                      n_iter = 20,
                      verbose = 0, 
                      n_jobs = 1
                     ):
    """
    Wrapper for Bayesian Optimization for hyperparameter tuning.
    
    Arguments:
    clf -- input classifier. May be of sklearn class.
    param_grid -- input parameter grid, in a dictionary format.
    X_train, X_test, y_train, y_test -- X and y data.
    beta -- float, beta < 1 favors more on NPV, while beta > 1 more on specificity
    threshold -- float, Threshold for classifying positive vs. negative classes.
    n_iter -- int, number of iterations to carry out Bayesian Optimization.
    verbose -- int, verbosity controller.
    n_jobs -- int, number of processes/threads being used.
    
    Returns:
    opt_rf -- The optimized classifier.
    """
        
    # RandomForest + Bayesian optimization
    opt_rf = BayesSearchCV(clf,
                           param_grid, 
                           n_iter = n_iter, verbose = verbose, 
                           scoring=make_scorer(adjusted_neg_Fbeta_score, 
                                               needs_proba=True, 
                                               beta=beta, 
                                               threshold=threshold),
                           n_jobs=n_jobs)

    # callback handler
    def on_step(optim_result):
        score = opt_rf.best_score_
        print("\t- best score: %s" % score)
        if score >= 0.99:
            print('* Interrupting...')
            return True

    opt_rf.fit(X_train, y_train.ravel(), callback=[on_step])
    
    y_prob = np.asarray(opt_rf.predict_proba(X_test))
    y_test = np.asarray(y_test)
    summarize_res(opt_rf, y_prob, y_test)
    
    return opt_rf