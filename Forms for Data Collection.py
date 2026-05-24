from flask import Flask, request, jsonify, send_from_directory
import os, json, random, datetime, csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from model import *

try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:
    raise RuntimeError('Please install scikit-learn: pip install scikit-learn') from exc

try:
    from scipy.io import arff
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = Flask(__name__, static_folder='.', static_url_path='')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'Sincerity importance',
    'Intelligence importance',
    'Fun importance',
    'Ambition importance',
    'Shared Interests importance'
]

DATA_FEATURES = [
    'sincere_important',
    'intelligence_important',
    'funny_important',
    'ambition_important',
    'shared_interests_important'
]

# Fields shown on each profile card (sourced from ARFF)
PROFILE_FIELDS = [
    'age',
    'pref_o_sincere',
    'pref_o_intelligence',
    'pref_o_funny',
    'pref_o_ambitious',
    'pref_o_shared_interests',
    'sincere',
    'intelligence',
    'funny',
    'ambition',
    # Per-interest ratings (1-10) used for compatibility display and SHAP training
    'sports',
    'tvsports',
    'exercise',
    'dining',
    'art',
    'hiking',
    'gaming',
    'clubbing',
    'reading',
    'tv',
    'theater',
    'movies',
    'concerts',
    'music',
    'shopping',
    'yoga',
    'decision_o',   # kept for internal use / model training; not displayed
]

TARGET_NAME = 'decision'
ARFF_FILE = 'speeddating.arff'

# File where participant responses are appended (one JSON object per line)
SUBMISSIONS_FILE = 'submissions.jsonl'

# Number of profiles per experiment block
BLOCK_SIZE = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_numeric(array):
    if array.dtype.kind in 'SU':
        return np.array(
            [float(x.decode('utf-8') if isinstance(x, bytes) else x) for x in array],
            dtype=float
        )
    return array.astype(float)


def load_arff_data():
    """Load the full ARFF dataset and return (X, y) for model training."""
    if not SCIPY_AVAILABLE:
        raise RuntimeError('Please install scipy: pip install scipy')

    file_path = os.path.join(os.path.dirname(__file__), ARFF_FILE)
    arff_file = arff.loadarff(file_path)
    df = pd.DataFrame(arff_file[0]) 
    for col in df.columns:
        if df[col].dtype != "float64" :

            encode = OrdinalEncoder()
            encode.fit(df[[col]])

            df[col] = encode.fit_transform(df[[col]])
    df.dropna(inplace = True)
    df = add_one_hot_encoding_on_interest(df)
    return df


def load_valid_profiles():
    """
    Load all ARFF rows that have complete values for the PROFILE_FIELDS.
    Returns a list of dicts suitable for JSON serialisation.
    """
    if not SCIPY_AVAILABLE:
        return []

    file_path = os.path.join(os.path.dirname(__file__), ARFF_FILE)
    raw_data, _ = arff.loadarff(file_path)

    profiles = []
    for i, row in enumerate(raw_data):
        ok = True
        vals = {'arff_row': i}
        for f in PROFILE_FIELDS:
            v = row[f]
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            try:
                v = float(v)
                if not np.isfinite(v):
                    ok = False
                    break
            except (ValueError, TypeError):
                ok = False
                break
            vals[f] = round(v, 2)
        if ok:
            profiles.append(vals)

    return profiles


PROFILE_SETS_DIR = 'profile_sets'
GENDER_MAP = {
    'men': 'male',
    'women': 'female',
}

def load_profile_csv_set(gender, suffix):
    file_name = f'{gender}_set_{suffix}.csv'
    file_path = os.path.join(os.path.dirname(__file__), PROFILE_SETS_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Missing profile set file: {file_name}')

    df = pd.read_csv(file_path, index_col=0)

    profiles = []
    for i, row in df.iterrows():
        profile = {
            'arff_row': int(i),
            'age': float(row['age']),
            'sincere': float(row['sincere']),
            'intelligence': float(row['intelligence']),
            'funny': float(row['funny']),
            'ambition': float(row['ambition']),
            'pref_o_sincere': 0.0,
            'pref_o_intelligence': 0.0,
            'pref_o_funny': 0.0,
            'pref_o_ambitious': 0.0,
            'pref_o_shared_interests': 0.0,
            'decision_o': None,
        }
        for col in df.columns:
            if col in profile:
                continue
            value = row[col]
            if isinstance(value, str) and value.strip() == '':
                continue
            try:
                profile[col] = float(value)
            except (ValueError, TypeError):
                profile[col] = value
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    try:
        model, X, acc = train_model(load_arff_data())
        return model, X, acc
    except Exception as e:
        # print missing columns for debugging
        app.logger.error(f'ARFF model build failed: {e}')
        return None, None, None
        



MODEL, TRAIN_X, ACC = build_model()         # TODO: log the train and test accuracy in the final dataset, but not this value since it is not personalised
EXPLAINER = None


# Pre-load all valid profiles once at startup
ALL_PROFILES = load_valid_profiles()
app.logger.info(f'Loaded {len(ALL_PROFILES)} valid profiles from ARFF.')

# ---------------------------------------------------------------------------
# SHAP explanation formatting  (unchanged from original)
# ---------------------------------------------------------------------------

def format_shap_explanation(feature_values, model_output, shap_values=None, base_value=None):
    if shap_values is None:
        feature_importance = feature_values
        explanation = ['SHAP is not available. Showing a fallback importance estimate.']
    else:
        feature_importance = shap_values
        explanation = ['SHAP explanation from your personalized model:']

    ranked = sorted(
        zip(FEATURE_NAMES, feature_importance),
        key=lambda item: abs(item[1]),
        reverse=True
    )
    for feature, value in ranked:
        explanation.append(f'• {feature}: {value:.3f}')

    explanation.append('')
    explanation.append(f'Model probability score: {model_output:.3f}')
    if base_value is not None:
        explanation.append(f'Model base value: {base_value:.3f}')

    return '\n'.join(explanation)


def prepare_input(payload):
    points = payload.get('pointAllocation', {})
    values = np.array([
        points.get('intelligence', 0),
        points.get('sincerity', 0),
        points.get('fun', 0),
        points.get('ambition', 0),
        points.get('shared', 0)
    ], dtype=float)
    return values.reshape(1, -1)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('.', 'data_collection_form.html')


@app.route('/api/profiles', methods=['GET'])
def api_profiles():
    """
    Return two gender-specific profile blocks of 30 each, based on the participant's
    dating preference. The request should include prefGender=men|women.

    The backend randomizes whether set A appears in block1 or block2 for each
    run, so the same preference group does not always see set A first.

    Response shape:
    {
      "blockOrder": "A_then_B" | "B_then_A",
      "block1": [ <30 profile objects> ],
      "block2": [ <30 profile objects> ]
    }
    """
    pref_gender = request.args.get('prefGender', '').lower()
    if pref_gender == 'men':
        gender = 'male'
    elif pref_gender == 'women':
        gender = 'female'
    else:
        gender = random.choice(['male', 'female'])

    first_tag, second_tag = ('A', 'B') if random.choice([True, False]) else ('B', 'A')
    block_order = 'A_then_B' if first_tag == 'A' else 'B_then_A'

    try:
        block1 = load_profile_csv_set(gender, first_tag)
        block2 = load_profile_csv_set(gender, second_tag)
    except Exception as e:
        app.logger.error(f'Profile load error: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500

    for i, p in enumerate(block1):
        p['profile_id'] = f'b1_{i:02d}'
    for i, p in enumerate(block2):
        p['profile_id'] = f'b2_{i:02d}'

    return jsonify({'blockOrder': block_order, 'block1': block1, 'block2': block2})


@app.route('/api/shap-explanation', methods=['POST'])
def api_model_explanation():
    """Existing SHAP explanation endpoint – unchanged logic."""
    try:
        normalized_coefficients = compute_normalized_coefficients(MODEL, TRAIN_X)

        return jsonify({'normalized_coefficients': normalized_coefficients})        # this was changed, and the data format here is might be wrong
    except Exception as e:
        app.logger.error(f'API error: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/submit', methods=['POST'])
def api_submit():
    """
    Receive and persist a complete participant submission.

    Expected JSON body:
    {
      "participantId": str,           # browser-generated UUID
      "timestamp": str,               # ISO timestamp from client
      "pointAllocation": {            # step 1
        "intelligence": int,
        "sincerity": int,
        "fun": int,
        "ambition": int,
        "shared": int
      },
      "interestRatings": {            # step 2 – per-interest 1-10 ratings
        "sports": int, "tvsports": int, "exercise": int, "dining": int,
        "art": int, "hiking": int, "gaming": int, "clubbing": int,
        "reading": int, "tv": int, "theater": int, "movies": int,
        "concerts": int, "music": int, "shopping": int, "yoga": int
      },
      "blockOrder": "A_then_B" | "B_then_A",
      "block1": {
        "profiles": [ <profile objects as returned by /api/profiles> ],
        "ratings":  { "<profile_id>": int (0-10), ... },
        "profileData": [              # one entry per profile; used for SHAP training
          {
            "profile_id": str,
            "arff_row": int,
            "rating": int,            # participant's 0-10 decision (training target)
            "profile_sincere": float,
            "profile_intelligence": float,
            "profile_funny": float,
            "profile_ambition": float,
            "profile_interests": { <interest_key>: float, ... },  # profile's 1-10 ratings
            "interest_diffs": { <interest_key>: float, ... },     # participant − profile
            "interest_sums":  { <interest_key>: float, ... },     # shared enthusiasm proxy
            "shown_top_matches": [{ key, participant, profile, diff }, ...],
            "shown_top_diffs":   [{ key, participant, profile, diff }, ...],
            "decision_o": float       # partner's original ARFF decision
          }, ...
        ]
      },
      "shapExplanation": {            # what was shown after block 1
        "explanation_text": str,
        "model_probability": float,
        "feature_values": [float]
      },
      "recognizeModel": "yes" | "partly" | "no",
      "block2": {
        "profiles": [ <profile objects> ],
        "ratings":  { "<profile_id>": int (0-10), ... },
        "profileData": [ ... ]        # same shape as block1.profileData
      },
      "modelInfluence": "yes" | "no" | "maybe"
    }

    Each submission is appended as a single JSON line to submissions.jsonl
    so it can be loaded later with pandas read_json(lines=True).

    Key columns for SHAP model construction (per profile row in profileData):
      Features: pointAllocation (5), interestRatings (16), interest_diffs (16),
                interest_sums (16), profile personality (4) → ~57 features
      Target:   rating (0-10, or binarised at threshold)
    """
    try:
        payload = request.get_json(force=True)

        # Add a server-side timestamp for extra reliability
        payload['server_timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'

        # Save submission as before
        submissions_path = os.path.join(os.path.dirname(__file__), SUBMISSIONS_FILE)
        with open(submissions_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')


        # --- Ensure block 1 and 2 profile data are DataFrames ---
        import pandas as pd
        block1_profile_data = payload.get('block1', {}).get('profileData', [])
        block2_profile_data = payload.get('block2', {}).get('profileData', [])
        if not isinstance(block1_profile_data, pd.DataFrame):
            df_block1 = pd.DataFrame(block1_profile_data)
        if not isinstance(block2_profile_data, pd.DataFrame):
            df_block2 = pd.DataFrame(block2_profile_data)

        # --- Train model on block 1 data ---
        if not block1_profile_data.empty:
            try:
                features = []
                targets = []
                interest_keys = [
                    'sports', 'tvsports', 'exercise', 'dining', 'art', 'hiking', 'gaming', 'clubbing',
                    'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga'
                ]
                for _, profile in block1_profile_data.iterrows():
                    # Defensive: skip if rating is missing
                    if 'rating' not in profile or pd.isnull(profile['rating']):
                        continue
                    row = []
                    # pointAllocation (5)
                    pa = payload.get('pointAllocation', {})
                    row.extend([
                        pa.get('intelligence', 0),
                        pa.get('sincerity', 0),
                        pa.get('fun', 0),
                        pa.get('ambition', 0),
                        pa.get('shared', 0)
                    ])
                    # interestRatings (16)
                    ir = payload.get('interestRatings', {})
                    row.extend([ir.get(k, 0) for k in interest_keys])
                    # interest_diffs (16)
                    row.extend([profile.get('interest_diffs', {}).get(k, 0) for k in interest_keys])
                    # interest_sums (16)
                    row.extend([profile.get('interest_sums', {}).get(k, 0) for k in interest_keys])
                    # profile personality (4)
                    row.extend([
                        profile.get('profile_sincere', 0),
                        profile.get('profile_intelligence', 0),
                        profile.get('profile_funny', 0),
                        profile.get('profile_ambition', 0)
                    ])
                    features.append(row)
                    targets.append(profile['rating'])

                if features and targets:
                    X_block1 = np.array(features, dtype=float)
                    y_block1 = np.array(targets, dtype=float)
                    # Binarize target at threshold 5 (as in original model)
                    y_block1_bin = (y_block1 >= 5).astype(int)
                    # Train model
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_block1, y_block1_bin)
                    acc = model.score(X_block1, y_block1_bin)
                    # Update global model
                    global MODEL, TRAIN_X, ACC
                    MODEL = model
                    TRAIN_X = X_block1
                    ACC = acc
                    app.logger.info(f"Model retrained on block 1 data. Accuracy: {acc:.3f}")
            except Exception as e:
                app.logger.error(f"Block 1 model training error: {e}", exc_info=True)

        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f'Submit error: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
