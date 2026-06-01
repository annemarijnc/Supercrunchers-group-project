# Supercrunchers Speed Dating Study

A Flask-based web application designed for behavioral studies on decision-making in speed dating. The application uses Machine Learning to model user preferences, provides interpretability through feature importance, and collects data for psychological research.

## Project Overview

This study explores how individuals rate potential partners and how well a model can predict and explain those ratings. Participants go through a questionaire where they provide demographics, set their dating preferences, and rate two blocks of profiles. Between blocks, the system provides an "explanation" of the participant's own decision-making process based on a model trained on their first set of ratings.


## Technical Stack

- **Backend:** Python / Flask
- **Data Science:** 
  - `scikit-learn`: Logistic Regression, scaling, and metrics.
  - `shap`: Interpretability and feature importance.
  - `pandas` & `numpy`: Data manipulation.
  - `scipy`: Correlation analysis.
- **Frontend:** HTML5, CSS3

## Project Structure

```text
├── app.py                  # Main Flask application and routing
├── model.py                # ML model training and evaluation logic
├── csv_model_values.py     # Coefficient and results post-processing
├── text_assign_session.py  # Test script for `app.py`
├── requirements.txt        # Project dependencies
├── templates/              # Templates for the UI
│   ├── base.html           # Shared layout and CSS
│   ├── welcome.html        # Consent and intro
│   ├── demographics.html   # User data collection
│   ├── preferences.html    # Dating preferences and interest sliders
│   ├── block1.html         # First round of profile ratings
│   ├── explanation.html    # Model results and feature importance
│   ├── block2.html         # Second round of profile ratings
│   ├── comparison.html     # ML model accuracy and influence feedback
│   └── ...                 # Other workflow steps
├── profile_sets/           # Demographically segmented profile data
└── data/                   # Raw speed dating datasets (ARFF format)
```

## Getting Started

### Prerequisites

- Python 3.x
- `pip` or `conda`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Supercrunchers-group-project
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
2. **Access the study:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Data Management

- Participant results are automatically appended to `results.csv` upon completion of the study.
