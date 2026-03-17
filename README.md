## Career Prediction System

This is a **career recommendation web application** that suggests suitable IT career paths for students based on their self‑reported skills, interests, and preferences.  
It uses **machine learning** (Decision Tree) to predict the best‑fit role and stores each prediction in a **SQLite** database for later analysis.

---

### Features

- **Interactive web app (Streamlit)**  
  - Clean UI with sliders and dropdowns for skills, habits, and preferences  
  - Instant prediction with progress bar and success animation

- **Smart career recommendations**
  - Predicts **best‑fit career option** for a student
  - Shows **Top‑3 recommended careers with probabilities**
  - Supports multiple career areas such as development, security, testing, cloud, etc.

- **Machine learning pipeline**
  - Data preprocessing and feature engineering on `mldata.csv`
  - Encodes categorical variables and builds training matrix
  - Trained **DecisionTreeClassifier** stored as `weights.pkl`
  - Uses the same feature encoding at prediction time

- **Database logging and analytics**
  - Saves every prediction in **SQLite** (`CBRSdata.db`, table `predictiontable`)
  - Stores: Name, Contact Number, Email, input profile, and predicted career
  - In‑app **analytics page**:
    - View recent predictions in a table
    - Bar chart of **most common predicted careers**

- **Basic model explainability**
  - Shows global **feature importance** (which inputs the model uses most)
  - Correlation heatmap for key numerical features

---

### Tech Stack

- **Language**
  - **Python 3**

- **Web / UI**
  - **Streamlit**

- **Machine Learning & Data**
  - **scikit‑learn** (Decision Tree)
  - **pandas**
  - **numpy**
  - **seaborn**, **matplotlib**

- **Backend / Persistence**
  - **SQLite** via `sqlite3`
  - Model persistence with **pickle** (`weights.pkl`)

---

### Project Structure (main files)

- `app.py` – main Streamlit app:
  - loads dataset and trained model
  - defines feature encoding and prediction logic
  - handles UI, prediction, plots, and analytics

- `db.py` – database helper:
  - creates `predictiontable` if not exists
  - inserts each prediction row (including predicted career)
  - helper functions to fetch data and role counts for analytics

- `data/mldata.csv` – training dataset  
- `weights.pkl` – trained Decision Tree model  
- `CBRSdata.db` – SQLite database with all user predictions  
- `assets/` – images used in the UI and preview

---

### Setup Instructions (Windows, Python + venv)

#### 1. Clone or download the project

Place the folder somewhere like:

```text
d:\Python\Career-pridiction\Career-Prediction-System-main
```

#### 2. Create and activate a virtual environment

In **PowerShell**:

```powershell
cd "d:\Python\Career-pridiction\Career-Prediction-System-main"

py -3 -m venv .venv
.\.venv\Scripts\activate
```

(If `py` does not work, use `python` instead.)

#### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If `matplotlib` or any other package is missing:

```powershell
pip install matplotlib seaborn scikit-learn streamlit pandas numpy flask flasgger
```

---

### Running the App

With the virtual environment activated:

```powershell
cd "d:\Python\Career-pridiction\Career-Prediction-System-main"
streamlit run app.py
```

Streamlit will open a browser window (or give you a local URL such as `http://localhost:8501`).

---

### How It Works (High‑Level)

1. **Data loading & preprocessing**
   - Reads `./data/mldata.csv`
   - Cleans and encodes categorical columns (e.g. workshops, certifications, interested subjects)
   - Creates numeric features such as:
     - binary encodings for Yes/No fields
     - ordinal encodings for “poor/medium/excellent”
     - dummy variables for “Management or Technical”, “Smart or Hard worker”

2. **Model training (offline, in notebook)**
   - Notebook `careerPredictionModel.ipynb`:
     - Builds feature matrix `df_train_x` and labels `df_train_y`
     - Splits into train/test
     - Trains a `DecisionTreeClassifier`
     - Evaluates accuracy and confusion matrix
     - Saves the trained model as `weights.pkl` using `pickle.dump`

3. **Prediction in the app**
   - `app.py` loads `weights.pkl`
   - Collects user inputs from the sidebar
   - Encodes them into the same 21‑feature vector used during training
   - Calls `regressor.predict_proba` to get probabilities
   - Displays:
     - main predicted career
     - Top‑3 careers with probabilities
     - global feature importance chart

4. **Database logging & analytics**
   - Every prediction call:
     - Ensures `predictiontable` exists
     - Inserts a row into `CBRSdata.db` with user details, inputs, and `Predicted_role`
   - Analytics section:
     - Shows a table of recent predictions from `predictiontable`
     - Aggregates counts per `Predicted_role` and renders a bar chart of most common careers

---

### Developed by

**Nikita Mishra**  
BSc IT – Final Year

