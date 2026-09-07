# Careermore AI

Careermore AI is a Streamlit-based career recommendation system for students and early-career learners. It uses a trained machine learning model to recommend suitable IT career paths from a user's skills, interests, work preferences, and learning habits.

The application also includes account authentication, prediction history, SQLite persistence, and community-level analytics.

## Features

- Interactive prediction form with skill ratings and preference selectors
- Career prediction using a persisted `DecisionTreeClassifier`
- Top 3 career recommendations with confidence scores
- User registration and login with bcrypt password hashing
- Personal prediction history for each logged-in user
- Community analytics showing the most frequently predicted roles
- Input validation for email addresses and phone numbers
- Automatic SQLite database and table initialization
- Responsive Streamlit UI with animations and visual feedback

## Technology Stack

- Python 3
- Streamlit and `streamlit-lottie`
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- SQLite through Python's built-in `sqlite3` module
- bcrypt for password hashing
- Pickle for loading the trained model

## Project Structure

```text
.
├── app.py                       # Streamlit application and prediction workflow
├── db.py                        # Authentication, persistence, and analytics helpers
├── data/
│   └── mldata.csv               # Dataset used for model features and encodings
├── weights.pkl                  # Trained model loaded by the application
├── CBRSdata.db                  # Local SQLite database, created/updated at runtime
├── careerPredictionModel.ipynb  # Model training and evaluation notebook
├── pythonFunctions/             # Supporting and experimental Python scripts
├── assets/                      # Images and other UI assets
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

- Python 3.9 or newer recommended
- PowerShell, Command Prompt, or a compatible terminal
- Internet access on first run if remote Lottie or image assets are used by the UI

## Installation

### 1. Create a virtual environment

From the project root:

```powershell
py -3 -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the `py` command is unavailable, use `python` instead.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

Make sure the virtual environment is active and run:

```powershell
streamlit run app.py
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Using the Application

1. Register with an email address and password, then log in.
2. Open **Prediction** from the sidebar.
3. Enter personal details and rate skills such as logical reasoning, coding, hackathons, and public speaking.
4. Select learning habits, interests, preferred books, certifications, workshops, and career area.
5. Click **Predict Your Career** to see the best match and the top 3 recommendations.
6. Open **User Profile** to review your previous predictions.
7. Open **Analytics** to view aggregated career prediction counts.

## How Prediction Works

The application prepares the input profile using the same transformations used during training:

- Yes/No responses are converted to binary values.
- `poor`, `medium`, and `excellent` values are converted to ordinal scores.
- Categorical fields are converted to numeric category codes.
- Management/Technical and Smart/Hard Worker values are represented using one-hot features.

The resulting 21-feature vector is passed to the model in `weights.pkl`. The application selects the class with the highest probability and displays the three highest-probability career options.

## Model Training

The training workflow is documented in `careerPredictionModel.ipynb`. In general, retraining involves:

1. Loading `data/mldata.csv`.
2. Applying the same preprocessing and feature encoding.
3. Training and evaluating a classifier.
4. Saving the trained model as `weights.pkl`.

After replacing `weights.pkl`, restart Streamlit so the application loads the updated model. The feature order and category mappings must remain compatible with `app.py`.

## Data and Privacy

Prediction inputs and account information are stored locally in `CBRSdata.db`. The database contains user identifiers, hashed passwords, profile inputs, and predicted roles. Do not commit real personal data or production credentials to a public repository.

For a fresh local installation, the app creates the required `users` and `predictiontable` tables automatically. Delete `CBRSdata.db` only when you intentionally want to remove local users and prediction history.

## Troubleshooting

**`FileNotFoundError: weights.pkl`**
Run the app from the project root and confirm that `weights.pkl` exists beside `app.py`.

**`FileNotFoundError: data/mldata.csv`**
Run `streamlit run app.py` from the project root so the relative data path resolves correctly.

**A package import fails**
Activate `.venv` and run `python -m pip install -r requirements.txt` again.

**The model gives unexpected results after retraining**
Verify that the new model uses the same feature order, category mappings, and target labels expected by `app.py`.

## License

See [LICENSE](LICENSE) for the project license.
Developed by Kshitij Pathak
