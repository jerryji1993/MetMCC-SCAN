import pickle as pkl
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

# class LogTransformer(BaseEstimator,TransformerMixin):
#     def __init__(self, constant=1, base='e'):
#         if base == 'e' or base == np.e:
#             self.log = np.log
#         elif base == '10' or base == 10:
#             self.log = np.log10
#         else:
#             base_log = np.log(base)
#             self.log = lambda x: np.log(x)/base_log
#         self.constant = constant
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         return self.log(X+self.constant)
    
class RoundTransformer(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.round()

def preprocess(df, num_cols, cat_cols, model_dir = None):
    # logTransformer = FunctionTransformer(np.log1p, validate=True)
    numeric_pipeline = make_pipeline(KNNImputer(), RoundTransformer())
    cat_pipeline     = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse=False))

    ct = [
            ('num', numeric_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
         ]

    preprocessor = Pipeline(steps=[
            ('col_transformers',ColumnTransformer(ct, remainder='passthrough'))
        ])
    

    df = preprocessor.fit_transform(df)
    
    if model_dir is not None:
        with open(model_dir+'/preprocessor.pkl', 'wb') as f:
            pkl.dump(preprocessor, f)
            
    return df