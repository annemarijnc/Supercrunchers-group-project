from flask import Flask, request, jsonify, send_from_directory
import os
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
TARGET_NAME = 'decision'
ARFF_FILE = 'speeddating.arff'


def safe_numeric(array):
    if array.dtype.kind in 'SU':
        return np.array([float(x.decode('utf-8') if isinstance(x, bytes) else x) for x in array], dtype=float)
    return array.astype(float)


def load_arff_data():
    if not SCIPY_AVAILABLE:
        raise RuntimeError('Please install scipy to load speeddating.arff: pip install scipy')

    file_path = os.path.join(os.path.dirname(__file__), ARFF_FILE)
    raw_data, meta = arff.loadarff(file_path)
    X = np.column_stack([safe_numeric(raw_data[field]) for field in DATA_FEATURES])
    y_raw = raw_data[TARGET_NAME]
    if y_raw.dtype.kind in 'SU':
        y = np.array([
            int(x.decode('utf-8') if isinstance(x, bytes) else x)
            for x in y_raw
        ], dtype=int)
    else:
        y = y_raw.astype(int)

    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    return X, y


def build_dummy_model():
    X = np.array([
        [20, 20, 20, 20, 20],
        [10, 40, 20, 10, 20],
        [5, 10, 60, 15, 10],
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
        if TRAIN_X is not None and len(TRAIN_X) > 0:
            sample_size = min(50, len(TRAIN_X))
            background = TRAIN_X[np.random.choice(len(TRAIN_X), sample_size, replace=False)]
        else:
            background = np.zeros((3, len(FEATURE_NAMES)))
        EXPLAINER = shap.Explainer(MODEL.predict_proba, background, feature_names=FEATURE_NAMES)
    except Exception:
        EXPLAINER = None


def format_shap_explanation(feature_values, model_output, shap_values=None, base_value=None):
    if shap_values is None:
        feature_importance = feature_values
        explanation = [
            'SHAP is not available in this environment. Showing a fallback importance estimate.'
        ]
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


@app.route('/')
def index():
    return send_from_directory('.', 'data_collection_form.html')


@app.route('/api/shap-explanation', methods=['POST'])
def api_shap_explanation():
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
                        if base_vals.ndim == 1 and len(base_vals) > 1:
                            base_value = float(base_vals[1])
                        elif base_vals.ndim == 0:
                            base_value = float(base_vals)
                    else:
                        base_value = float(base_vals)
                
                explanation_text = format_shap_explanation(X[0], proba, shap_values_for_positive, base_value)
            except Exception as e:
                app.logger.error(f'SHAP error: {str(e)}', exc_info=True)
                weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
                values = X[0]
                fallback_values = ((values - np.mean(values)) * weights).tolist()
                explanation_text = format_shap_explanation(values, proba, fallback_values, None)
        else:
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
            values = X[0]
            fallback_values = ((values - np.mean(values)) * weights).tolist()
            explanation_text = format_shap_explanation(values, proba, fallback_values, None)

        return jsonify({
            'success': True,
            'model_probability': float(proba),
            'explanation_text': explanation_text,
            'feature_values': X[0].tolist()
        })
    except Exception as e:
        app.logger.error(f'API error: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
