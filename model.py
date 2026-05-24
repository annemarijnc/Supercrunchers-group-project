import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
import numpy as np

REQUIRED_COLUMNS = ['age', 'd_age', 'pref_o_sincere', 'pref_o_intelligence',
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

INTEREST_COLUMNS = ['d_sports', 'd_tvsports', 'd_exercise', 'd_dining', 'd_museums', 'd_art', 'd_hiking', 'd_gaming',
    'd_clubbing', 'd_reading', 'd_tv', 'd_theater', 'd_movies', 'd_concerts', 'd_music', 'd_shopping',
    'd_yoga']

def rating_to_decision(rating, highest_rating = 10):
    rating = int(rating)
    if rating > highest_rating / 2:
        return 1
    else:
        return 0


def encode_features(df):
    for col in df.columns:
        if df[col].dtype != "float64" :

            encode = OrdinalEncoder()
            encode.fit(df[[col]])

            df[col] = encode.fit_transform(df[[col]])
    df.dropna(inplace = True)
    return df

def add_one_hot_encoding_on_interest(df):
    for col in INTEREST_COLUMNS:
        like_name = col.replace('d_', 'both_like_')
        dislike_name = col.replace('d_', 'both_dislike_')
        df[like_name] = 0
        df[dislike_name] = 0

    for row in df.itertuples():
        # make list of all values in interest_cols for this row
        interests = [getattr(row, col) for col in INTEREST_COLUMNS]
        # get index of top 2 and bottom 2 values
        top_2_idx = sorted(range(len(interests)), key=lambda i: interests[i], reverse=True)[:2]
        bottom_2_idx = sorted(range(len(interests)), key=lambda i: interests[i])
        # get names of top_2_idx and bottom_2_idx
        top_2_cols = [INTEREST_COLUMNS[i] for i in top_2_idx]
        bottom_2_cols = [INTEREST_COLUMNS[i] for i in bottom_2_idx]
        # set like_name to 1 for top_2_cols and dislike_name to 1 for bottom_2_cols
        for col in top_2_cols:
            like_name = col.replace('d_', 'both_like_')
            df.at[row.Index, like_name] = 1
        for col in bottom_2_cols:
            dislike_name = col.replace('d_', 'both_dislike_')
            df.at[row.Index, dislike_name] = 1

    # remove original interest columns
    df.drop(INTEREST_COLUMNS, axis=1, inplace=True)
    return df

def split_data_for_training(df):
    X = df.drop('decision_o', axis=1)
    print("Columns in X: ", X.columns)
    print("Shape of X: ", X.shape)
    print("Values of X: ", X.head())
    # print column that contain null values    
    print("Columns with null values: ", X.columns[X.isnull().any()])
    y = df['decision_o']
    # X = StandardScaler().fit_transform(X)
    return X, y

def train_model(df):
    # save df to csv for debugging
    df.to_csv('df_for_training.csv', index=False)
    df = encode_features(df)
    # if not all(col in df.columns for col in REQUIRED_COLUMNS):
    #     print("Columns in df: ", df.columns)
    #     raise ValueError(f"DataFrame does not contain the required columns. Check if the one-hot encoding is applied.")
    
    X, y = split_data_for_training(df)
    model = LogisticRegression(max_iter=500)
    model.fit(df, y)
    acc = model.score(df, y)
    print("Accuracy (on train set): ", acc)
    return model, X, acc

def evaluate_model(model, X, y):
    acc = model.score(X, y)
    print("Accuracy (on test set): ", acc)
    return model, X, acc

def compute_normalized_coefficients(model, X):
    # TODO: add age and d_age
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

    # normalize coeficients between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_coeficients = scaler.fit_transform(np.array(coeficients).reshape(-1, 1)).flatten()
    
    normalized_coeficients = []

    for idx in range(len(features)):
        normalized_coeficient = scaled_coeficients[idx] / scaled_coeficients.sum() * 100
        normalized_coeficients.append(float(normalized_coeficient))
    print(f'Normalized coefficients: {normalized_coeficients}')

    return normalized_coeficients