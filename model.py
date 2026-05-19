import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

REQUIRED_COLUMNS = ['age', 'd_age', 'gender', 'pref_o_sincere', 'pref_o_intelligence',
       'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests',
       'sincere', 'intelligence', 'funny', 'ambition', 'interests_correlate',
       'decision_o', 'both_like_sports', 'both_dislike_sports',
       'both_like_tvsports', 'both_dislike_tvsports', 'both_like_exercise',
       'both_dislike_exercise', 'both_like_dining', 'both_dislike_dining',
       'both_like_museums', 'both_dislike_museums', 'both_like_art',
       'both_dislike_art', 'both_like_hiking', 'both_dislike_hiking',
       'both_like_gaming', 'both_dislike_gaming', 'both_like_clubbing',
       'both_dislike_clubbing', 'both_like_reading', 'both_dislike_reading',
       'both_like_tv', 'both_dislike_tv', 'both_like_theater',
       'both_dislike_theater', 'both_like_movies', 'both_dislike_movies',
       'both_like_concerts', 'both_dislike_concerts', 'both_like_music',
       'both_dislike_music', 'both_like_shopping', 'both_dislike_shopping',
       'both_like_yoga', 'both_dislike_yoga']

def split_data_for_training(df):
    X = df.drop('decision_o', axis=1)
    y = df['decision_o']
    X.drop('gender', axis=1, inplace=True)
    X = StandardScaler().fit_transform(X)
    return X, y

def train_model(df):
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError("DataFrame does not contain the required columns")
    
    X, y = split_data_for_training(df)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    acc = model.score(X, y)
    print("Accuracy (on train set): ", acc)
    return model, acc

def compute_normalized_coefficients(model, X):
    features = ['sincere',
            'intelligence',
            'funny',
            'ambition',
            'interests_correlate'      # tells us level of shared interest
            ]
    features_idx = [X.columns.get_loc(feature) for feature in features]   
    coeficients = []

    for idx in features_idx:
        coeficients.append(float(model.coef_[0][idx]))
        print(f'Coefficient for feature {X.columns[idx]}: {model.coef_[0][idx]:.2f}')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    coeficients = scaler.fit_transform(np.array(coeficients).reshape(-1, 1)).flatten()
    print(f'Normalized coefficients: {coeficients}')

    return coeficients