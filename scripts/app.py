# general modules
import os
from copy import deepcopy
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from load_css import local_css

# imblearn model
from imblearn.ensemble import BalancedRandomForestClassifier



MODEL_DIR = '../model/balanced_rf_knnimputed_binned.pkl'
DATA_DIR = '../data/Merkel_NCDB_clean_selectedVars_imputed.pkl'
IMPUTER_DIR = '../model/knn_imputer.pkl'

@st.cache(allow_output_mutation=True)
def load_model():
    with open(MODEL_DIR, 'rb') as f:
        model = pkl.load(f)
    return model

@st.cache(allow_output_mutation=True)
def load_imputer():
    with open(IMPUTER_DIR, 'rb') as f:
        imputer = pkl.load(f)
    return imputer

@st.cache
def load_data():
    with open(DATA_DIR, 'rb') as f:
        full_data = pkl.load(f)
    return full_data

def get_sidebar_input():
    # demographic
    age = st.sidebar.slider('Age of patient: ', min_value= 20, max_value=100, value=60, step=1)
    gender = st.sidebar.radio(label="Gender of patient: ", options = ('Male', 'Female'))

    # primary site
    primary_site = st.sidebar.radio(label="Site of the primary tumor: ", options = ('Head and Neck',
                                                                                    'Trunk',
                                                                                    'Extremity',
                                                                                    'Other'))
    # tumor size
    tumor_size_known = st.sidebar.checkbox('Have information about diameter of primary tumor?')
    if tumor_size_known:
        tumor_size = st.sidebar.number_input('''Largest dimension of diameter of the primary tumor in 
                        millimeters (mm):''', min_value = 0, max_value=995, value=1,step = 1)
    else:
        tumor_size = 999

    # depth
    depth_known = st.sidebar.checkbox('Have information about thickness/depth of primary tumor?')
    if depth_known:
        depth = st.sidebar.number_input('''Measured thickness (depth) of primary tumor in 
                        millimeters (mm):''', min_value = 0, max_value=980, value=1,step = 1)
    else:
        depth = 999

    # lymph vascular invasion
    LVI = st.sidebar.radio(label="Tumor cells present in lymphatic channels/blood vessels within the primary tumor: ", 
                           options = ('Yes',
                                      'No',
                                      'Unknown/Not applicable'))

    # tumor infiltrating lymphocytes
    TIL = st.sidebar.select_slider(label="Tumor infiltrating lymphocytes (TILs) present within primary tumor:", 
                           options = ('None',
                                      'Weakly present',
                                      'Strongly present',
                                      'Present (no strength)',
                                      'Unknown'))

    # immune suppression
    immune = st.sidebar.radio(label="Profound immune suppression found: ", 
                           options = ('Positive',
                                      'Negative',
                                      'Unknown/Not applicable'))

    # growth pattern
    growth = st.sidebar.radio(label="Growth pattern of primary tumor: ", 
                           options = ('Circumscribed nodular',
                                      'Diffusely infiltrative',
                                      'Unknown/Not applicable'))

    # tumor base transection
    transection = st.sidebar.radio(label="Tumor base is transected: ", 
                           options = ('Transected',
                                      'Not transected',
                                      'Not found',
                                      'Unknown/Not applicable'))
    inputs = {
                'AGE': age, 'TUMOR_SIZE': tumor_size, 'DEPTH': depth, 'SEX': gender,
                'PRIMARY_SITE': primary_site, 'LYMPH_VASCULAR_INVASION': LVI,
                'TUMOR_INFILTRATING_LYMPHOCYTES': TIL, 'IMMUNE_SUPPRESSION': immune,
                'GROWTH_PATTERN': growth, 'TUMOR_BASE_TRANSECTION': transection
             }
    
    return inputs

def rename_categories(df):
    df = df.replace('Unknown/Not applicable', 'Unknown')
    df['TUMOR_INFILTRATING_LYMPHOCYTES'] = df['TUMOR_INFILTRATING_LYMPHOCYTES'].replace(['None',
                                      'Weakly present',
                                      'Strongly present',
                                      'Present (no strength)',
                                      'Unknown'],
                                    ['Negative','Weak','Strong','Present','Unknown'])
    df = df.replace(' ','_', regex=True)
    return df

@st.cache
def align_data(df, new_df):
    df = pd.get_dummies(df)
    new_df = pd.get_dummies(new_df)
    df, new_df = df.align(new_df, join='left', axis=1)
    new_df.drop('SLNB', axis=1, inplace=True)
    new_df = new_df.fillna(0)
    new_df = new_df.replace([999],[np.NaN])
    return new_df

def impute(imputer, df):
    return imputer.transform(df)
    

def main():
    st.title('MetMCC-SCAN')
    st.header('A Predictor for Metastasis of Merkel Cell Carcinoma')
    st.subheader("*Reducing Unnecessary Sentinel Lymph Node Biopsies*")
    inputs = pd.DataFrame(get_sidebar_input(), index=[0])
    inputs = rename_categories(inputs)
    
    ### TODO: change this
    full_data = load_data()
    inputs = align_data(full_data, inputs)
    
#     st.dataframe(inputs)
    if st.button("Predict"):
        # load imputer
        imputer = load_imputer()
        inputs = impute(imputer, inputs)
        # load model
        model = load_model()
        y_prob = np.asarray(model.predict_proba(inputs))
        y_pred = np.asarray(model.predict(inputs))

        # output prediction
        st.write("Probability of having a positive biopsy:", y_prob[:,1][0])
        local_css("style.css")
        if y_pred[0] == 0:

            t = '''<div>The patient is likely to have a 
                    <span class='highlight blue'>negative </span> 
                    Sentinel Lymph Node Biopsy result.
                   </div>
                '''

        else:
            t = '''<div>The patient is likely to have a 
                    <span class='highlight red'>positive </span>  
                    Sentinel Lymph Node Biopsy result.
                   </div>
                '''

            
        st.markdown(t, unsafe_allow_html=True)
        st.markdown("""<br>""", unsafe_allow_html=True)
        
    image = Image.open('../image/lymph_nodes.jpg')
    st.image(image, use_column_width=True)
    
#     st.markdown("""<br>""", unsafe_allow_html=True)
#     st.markdown("""<br>""", unsafe_allow_html=True)
#     st.markdown("""<br>""", unsafe_allow_html=True)
#     st.markdown("""<iframe src="https://docs.google.com/presentation/d/1--eW4tCH3lwxLpfyjghiqK3en7VOY016BZvjH87k4mw/embed?start=false&loop=false&delayms=10000" frameborder="0" width="480" height="299" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>""", unsafe_allow_html=True
#     )

if __name__ == "__main__":
    main()
