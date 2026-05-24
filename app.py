import os
import csv
import random
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from model import *

app = Flask(__name__)
app.secret_key = os.urandom(24)

MODEL = None
X = None

# Define the sequence of steps
STEPS = [
    'welcome',
    'demographics',
    'preferences',
    'instructions',
    'block1',
    'explanation',
    'block2',
    'comparison',
    'debriefing',
    'thanks'
]
INTEREST_FIELDS = [
            'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming',
            'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga'
        ]

CSV_FILE = 'results.csv'

SESSION_KEYS = [
    'timestamp', 'gender', 'age', 'preference', 'status',
    'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests'
    ] + INTEREST_FIELDS + [
    'block1_set', 'id_block1', 'ratings_block1', 'train_acc', 'agreement', 'trust',
    'id_block2', 'ratings_block2', 'test_acc', 'influence', 'comments'
]



# Returns a dictionary with the specified range of SESSION_KEYS columns using current session values
def get_session_as_dict(range_start=0, range_end=None):
    if range_end is None:
        range_end = len(SESSION_KEYS)
    return {key: session.get(key) for key in SESSION_KEYS[range_start:range_end]}

def build_full_df(dct, df_path, first_block = True):
    # Step 1: Load profiles
    df = pd.read_csv('profile_sets/' + df_path)
    print("Loaded DataFrame:")
    print(df.head())

    # Step 2: Get ids and ratings from session dict
    ids = dct['id_block1'] if first_block else dct['id_block2']
    ratings = dct['ratings_block1'] if first_block else dct['ratings_block2']
    print("IDs from session:", ids)
    print("Ratings from session:", ratings)

    # check if ids from df and dct match
    df_ids = set(df['id'].astype(str))
    session_ids = set(ids)
    print("IDs in DataFrame:", df_ids)
    print("IDs in session:", session_ids)
    if not session_ids.issubset(df_ids):
        print("Warning: Some IDs from session are not in DataFrame!")
        print("IDs in session not in DataFrame:", session_ids - df_ids)

    # Step 3: Assign decision_o for rated profiles
    df['decision_o'] = None
    for id, rating in zip(ids, ratings):
        try:
            id_int = int(id)
            dec = rating_to_decision(rating)
            df.loc[df['id'] == id_int, 'decision_o'] = dec
            print(f"Assigned decision_o={dec} for id={id_int}")
        except Exception as e:
            print(f"Error assigning decision_o for id={id}, rating={rating}: {e}")

    print("DataFrame after assigning decision_o:")
    print(df[['id', 'decision_o']].head(10))
    print("decision_o value counts:")
    print(df['decision_o'].value_counts(dropna=False))
    # check if there are at least 2 different classes in decision_o
    if df['decision_o'].nunique(dropna=True) < 2:
        print("Warning: Not enough classes in decision_o for training!")
        print("Unique values in decision_o:", df['decision_o'].unique())

    # Step 4: Add user preference columns
    df['pref_o_sincere'] = dct['pref_o_sincere']
    df['pref_o_intelligence'] = dct['pref_o_intelligence']
    df['pref_o_funny'] = dct['pref_o_funny']
    df['pref_o_ambitious'] = dct['pref_o_ambitious']
    df['pref_o_shared_interests'] = dct['pref_o_shared_interests']
    df['d_age'] = df['age'].astype(int) - int(dct['age'])

    # Step 5: Add interest difference columns
    for interest in INTEREST_FIELDS:
        df['d_' + interest] = df[interest].astype(float) - float(dct[interest])

    # Step 6: Calculate interest correlation
    correlations = [df[col].corr(df['decision_o']) for col in [f'd_{interest}' for interest in INTEREST_FIELDS]]
    df['interests_correlate'] = np.mean([c for c in correlations if not np.isnan(c)])

    # Step 7: Drop original interest columns
    df.drop(INTEREST_FIELDS, axis=1, inplace=True)

    # Step 8: One-hot encoding
    df = add_one_hot_encoding_on_interest(df)

    # Step 9: Filter to only rows with a decision (rated by user)
    df_filtered = df[df['decision_o'].notnull()]
    print("Filtered DataFrame (only rated profiles):")
    print(df_filtered[['id', 'decision_o']].head(10))
    print("Filtered decision_o value counts:")
    print(df_filtered['decision_o'].value_counts(dropna=False))

    print(f'Columns in df after one-hot encoding: {df.columns.tolist()}')
    return df_filtered


def execute_model(evaluate=False):
    global MODEL, X
    dct = get_session_as_dict()
    print(dct)
    gender = 'male' if dct['preference'] == 'Men' else 'female'
    df_path = gender + '_set_' + dct['block1_set'] + '.csv'
    full_df = build_full_df(dct, df_path)
    print("Columns in full_df: ", full_df.columns)
    print("Shape of full_df: ", full_df.shape)
    print("Values of full_df: ", full_df.head())
    print("Columns with null values: ", full_df.columns[full_df.isnull().any()])

    if evaluate:                    # check if this is working correctly
        y = full_df['decision_o']
        _, _, test_acc = evaluate_model(MODEL, X, y)
        return test_acc
    else:
        MODEL, X, train_acc = train_model(full_df)
        return MODEL, X, train_acc

INTERESTS = [
    'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 
    'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 
    'movies', 'concerts', 'music', 'shopping', 'yoga'
]

def get_current_step():
    return session.get('current_step', 0)

def set_step(step_index):
    current = get_current_step()
    if step_index > current:
        session['current_step'] = step_index

def save_to_csv(data):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_KEYS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def load_profiles(preference, set_name):
    gender_prefix = 'male' if preference == 'Men' else 'female'
    filename = f'profile_sets/{gender_prefix}_set_{set_name}.csv'
    profiles = []
    if os.path.exists(filename):
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profiles.append(row)
    return profiles

def get_profile_data(row):
    age_raw = row.get('age', 'N/A')
    try:
        age = int(float(age_raw)) if age_raw != 'N/A' else 'N/A'
    except ValueError:
        age = age_raw
    
    # Convert interest values to floats for sorting
    scores = []
    for interest in INTERESTS:
        try:
            val = float(row.get(interest, 0))
        except ValueError:
            val = 0
        scores.append((interest, val))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    top_2 = [s[0] for s in scores[:2]]
    bottom_2 = [s[0] for s in scores[-2:]]
    
    return {
        'age': age,
        'likes': top_2,
        'dislikes': bottom_2
    }

@app.route('/')
def index():
    session.clear() # Clear session on start
    session['current_step'] = 0
    return redirect(url_for('welcome'))

@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    if get_current_step() > 0:
        return redirect(url_for(STEPS[get_current_step()]))
    
    if request.method == 'POST':
        consent = request.form.get('consent')
        if consent == 'yes':
            session['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            set_step(1)
            return redirect(url_for('demographics'))
        else:
            return "Please inform the teacher as you did not consent."
    
    return render_template('welcome.html')

@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    step_idx = 1
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    if request.method == 'POST':
        session['gender'] = request.form.get('gender')
        session['age'] = request.form.get('age')
        session['preference'] = request.form.get('preference')
        session['status'] = request.form.get('status')
        set_step(2)
        return redirect(url_for('preferences'))
    
    return render_template('demographics.html')

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    step_idx = 2
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))

    if request.method == 'POST':
        # Retrieve the 5 numeric values from the form
        session['pref_o_sincere'] = request.form.get('attr1')
        session['pref_o_intelligence'] = request.form.get('attr2')
        session['pref_o_funny'] = request.form.get('attr3')
        session['pref_o_ambitious'] = request.form.get('attr4')
        session['pref_o_shared_interests'] = request.form.get('attr5')
        for interest in INTEREST_FIELDS:
            session[interest] = request.form.get(interest)
        set_step(3)
        return redirect(url_for('instructions'))    
    
    return render_template('preferences.html')

@app.route('/instructions', methods=['GET', 'POST'])
def instructions():
    step_idx = 3
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    if request.method == 'POST':
        # Randomly assign sets for block 1 and block 2
        set1 = random.choice(['A', 'B'])
        set2 = 'B' if set1 == 'A' else 'A'
        session['block1_set'] = set1
        session['block2_set'] = set2
        session['current_profile_idx'] = 0
        session['ratings_block1'] = []
        session['ratings_block2'] = []
        session['id_block1'] = []
        session['id_block2'] = []
        set_step(4)
        return redirect(url_for('block1'))
    
    return render_template('instructions.html')

@app.route('/block1', methods=['GET', 'POST'])
def block1():
    step_idx = 4
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    profiles = load_profiles(session.get('preference'), session.get('block1_set'))
    current_idx = session.get('current_profile_idx', 0)
    
    if current_idx >= len(profiles):
        session['current_profile_idx'] = 0 # Reset for block 2
        set_step(5)
        return redirect(url_for('explanation'))
    
    if request.method == 'POST':
        rating = request.form.get('rating')
        ratings = session.get('ratings_block1', [])
        ratings.append(rating)
        session['ratings_block1'] = ratings
        # Track profile ID (use 'iid' if present, else row index)
        id_block1 = session.get('id_block1', [])
        profile_row = profiles[current_idx]
        profile_id = profile_row.get('iid', str(current_idx))
        id_block1.append(str(profile_id))
        session['id_block1'] = id_block1
        session['current_profile_idx'] = current_idx + 1
        return redirect(url_for('block1'))
    profile_data = get_profile_data(profiles[current_idx])
    return render_template('block1.html', 
                           profile=profile_data, 
                           current_idx=current_idx, 
                           total_profiles=len(profiles))

@app.route('/explanation', methods=['GET', 'POST'])
@app.route('/explanation', methods=['GET', 'POST'])

def explanation():
    step_idx = 5
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))


    MODEL, X, train_acc = execute_model()
    session['train_acc'] = train_acc
    normalized_coeficients = compute_normalized_coefficients(MODEL, X)
    bar_values = normalized_coeficients
    bar_labels = ['Sincere', 'Intelligence', 'Fun', 'Ambition', 'Shared Interest']
    zipped_bars = zip(bar_labels, bar_values)

    if request.method == 'POST':
        session['agreement'] = request.form.get('agreement')
        session['trust'] = request.form.get('trust')
        set_step(6)
        return redirect(url_for('block2'))

    return render_template('explanation.html', zipped_bars=zipped_bars)

@app.route('/block2', methods=['GET', 'POST'])
def block2():
    step_idx = 6
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    profiles = load_profiles(session.get('preference'), session.get('block2_set'))
    current_idx = session.get('current_profile_idx', 0)
    
    if current_idx >= len(profiles):
        set_step(7)
        return redirect(url_for('comparison'))
    
    if request.method == 'POST':
        rating = request.form.get('rating')
        ratings = session.get('ratings_block2', [])
        ratings.append(rating)
        session['ratings_block2'] = ratings
        # Track profile ID (use 'iid' if present, else row index)
        id_block2 = session.get('id_block2', [])
        profile_row = profiles[current_idx]
        profile_id = profile_row.get('iid', str(current_idx))
        id_block2.append(str(profile_id))
        session['id_block2'] = id_block2
        session['current_profile_idx'] = current_idx + 1
        return redirect(url_for('block2'))
    profile_data = get_profile_data(profiles[current_idx])
    return render_template('block2.html', 
                           profile=profile_data, 
                           current_idx=current_idx, 
                           total_profiles=len(profiles))

@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    print("Session data at comparison step:", get_session_as_dict())
    step_idx = 7

    test_acc = execute_model(evaluate=True)
    session['test_acc'] = test_acc

    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    if request.method == 'POST':
        session['influence'] = request.form.get('influence')
        set_step(8)
        return redirect(url_for('debriefing'))
    
    return render_template('comparison.html')

@app.route('/debriefing', methods=['GET', 'POST'])
def debriefing():
    step_idx = 8
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    
    if request.method == 'POST':
        session['comments'] = request.form.get('comments')
        
        # Collect all data and save to CSV
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gender': session.get('gender'),
            'age': session.get('age'),
            'preference': session.get('preference'),
            'status': session.get('status'),
            'pref_o_sincere': session.get('pref_o_sincere'),
            'pref_o_intelligence': session.get('pref_o_intelligence'),
            'pref_o_funny': session.get('pref_o_funny'),
            'pref_o_ambitious': session.get('pref_o_ambitious'),
            'pref_o_shared_interests': session.get('pref_o_shared_interests'),
        }
        # Add all INTEREST_FIELDS to data
        for field in INTEREST_FIELDS:
            data[field] = session.get(field)
        data.update({
            'block1_set': session.get('block1_set'),
            'id_block1': ';'.join(session.get('id_block1', [])),
            'ratings_block1': ';'.join(session.get('ratings_block1', [])),
            'train_acc': session.get('train_acc'),
            'agreement': session.get('agreement'),
            'trust': session.get('trust'),
            'id_block2': ';'.join(session.get('id_block2', [])),
            'ratings_block2': ';'.join(session.get('ratings_block2', [])),
            'test_acc': session.get('test_acc'),
            'influence': session.get('influence'),
            'comments': session.get('comments')
        })
        save_to_csv(data)
        
        set_step(9)
        return redirect(url_for('thanks'))
    
    return render_template('debriefing.html')

@app.route('/thanks')
def thanks():
    step_idx = 9
    if get_current_step() != step_idx:
        return redirect(url_for(STEPS[get_current_step()]))
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
