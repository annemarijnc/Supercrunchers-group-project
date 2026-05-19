from flask import Flask, request, jsonify, send_from_directory
import os, json, random, datetime
import numpy as np

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
    'Intelligence importance',
    'Sincerity importance',
    'Fun importance',
    'Ambition importance',
    'Shared Interests importance'
]

DATA_FEATURES = [
    'intellicence_important',
    'sincere_important',
    'funny_important',
    'ambtition_important',
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
    raw_data, meta = arff.loadarff(file_path)

    X = np.column_stack([safe_numeric(raw_data[field]) for field in DATA_FEATURES])
    y_raw = raw_data[TARGET_NAME]
    if y_raw.dtype.kind in 'SU':
        y = np.array(
            [int(x.decode('utf-8') if isinstance(x, bytes) else x) for x in y_raw],
            dtype=int
        )
    else:
        y = y_raw.astype(int)

    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_dummy_model():
    X = np.array([
        [20, 20, 20, 20, 20],
        [10, 40, 20, 10, 20],
        [5,  10, 60, 15, 10],
        [15, 15, 20, 40, 10],
        [25, 15, 25, 25, 10]
    ], dtype=float)
    y = np.array([1, 1, 0, 0, 1], dtype=int)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    return model, X


def build_model():
    if SCIPY_AVAILABLE:
        try:
            X, y = load_arff_data()
            if len(np.unique(y)) < 2:
                raise ValueError('ARFF target contains only one class.')
            model = LogisticRegression(solver='liblinear', random_state=42)
            model.fit(X, y)
            return model, X
        except Exception as e:
            app.logger.error(f'ARFF model build failed: {e}', exc_info=True)
    return build_dummy_model()


MODEL, TRAIN_X = build_model()
EXPLAINER = None

if SHAP_AVAILABLE:
    try:
        sample_size = min(50, len(TRAIN_X))
        background = TRAIN_X[np.random.choice(len(TRAIN_X), sample_size, replace=False)]
        EXPLAINER = shap.Explainer(MODEL.predict_proba, background, feature_names=FEATURE_NAMES)
    except Exception:
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
    Return 60 randomly sampled profiles split into two blocks of 30.
    Each call produces a fresh random sample so every participant sees
    a different (randomised) set.

    Response shape:
    {
      "block1": [ <30 profile objects> ],
      "block2": [ <30 profile objects> ]
    }

    Each profile object:
    {
      "arff_row": int,          # original row index – useful for tracing back
      "profile_id": str,        # unique id like "p_0042"
      "age": float,
      "pref_o_sincere": float,          # partner's stated preference weight (0-100 pts)
      "pref_o_intelligence": float,
      "pref_o_funny": float,
      "pref_o_ambitious": float,
      "pref_o_shared_interests": float,
      "sincere": float,                 # self-rating (1-10)
      "intelligence": float,
      "funny": float,
      "ambition": float,
      "decision_o": float               # partner's actual decision (0/1) – for model training
    }
    """
    if len(ALL_PROFILES) < BLOCK_SIZE * 2:
        return jsonify({'error': 'Not enough valid profiles in dataset.'}), 500

    sampled = random.sample(ALL_PROFILES, BLOCK_SIZE * 2)

    # Randomly assign which 30 go to block 1 vs block 2
    random.shuffle(sampled)
    block1 = sampled[:BLOCK_SIZE]
    block2 = sampled[BLOCK_SIZE:]

    # Give each profile a stable id within this session
    for i, p in enumerate(block1):
        p['profile_id'] = f'b1_{i:02d}'
    for i, p in enumerate(block2):
        p['profile_id'] = f'b2_{i:02d}'

    return jsonify({'block1': block1, 'block2': block2})


@app.route('/api/shap-explanation', methods=['POST'])
def api_shap_explanation():
    """Existing SHAP explanation endpoint – unchanged logic."""
    try:
        payload = request.get_json(force=True)
        X = prepare_input(payload)
        proba = MODEL.predict_proba(X)[0, 1]

        if SHAP_AVAILABLE and EXPLAINER is not None:
            try:
                shap_result = EXPLAINER(X)
                shap_vals = shap_result.values
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    shap_values_for_positive = shap_vals[0, 1, :].tolist()
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                    shap_values_for_positive = shap_vals[0].tolist()
                else:
                    shap_values_for_positive = shap_vals[0].tolist()

                base_value = None
                if hasattr(shap_result, 'base_values'):
                    base_vals = shap_result.base_values
                    if isinstance(base_vals, np.ndarray):
                        base_value = float(base_vals[1]) if base_vals.ndim == 1 and len(base_vals) > 1 else float(base_vals)
                    else:
                        base_value = float(base_vals)

                explanation_text = format_shap_explanation(X[0], proba, shap_values_for_positive, base_value)
            except Exception as e:
                app.logger.error(f'SHAP error: {e}', exc_info=True)
                weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                fallback = ((X[0] - np.mean(X[0])) * weights).tolist()
                explanation_text = format_shap_explanation(X[0], proba, fallback, None)
        else:
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            fallback = ((X[0] - np.mean(X[0])) * weights).tolist()
            explanation_text = format_shap_explanation(X[0], proba, fallback, None)

        return jsonify({
            'success': True,
            'model_probability': float(proba),
            'explanation_text': explanation_text,
            'feature_values': X[0].tolist()
        })
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

        submissions_path = os.path.join(os.path.dirname(__file__), SUBMISSIONS_FILE)
        with open(submissions_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')

        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f'Submit error: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
