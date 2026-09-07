import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import streamlit as st
import re
import requests
from streamlit_lottie import st_lottie
from db import *

# Initialize DB
create_table()

# Load Model
pickleFile=open("weights.pkl","rb")
regressor=pickle.load(pickleFile)

# Load Data for encodings
df = pd.read_csv('./data/mldata.csv')
df['workshops'] = df['workshops'].replace(['testing'],'Testing')

# Encodings setup (extracted from original script to keep it functional)
cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    df = df.replace({i: {"yes": 1, "no": 0}})

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    df = df.replace({i: {"poor": 0, "medium": 1, "excellent": 2}})

category_cols = ['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])

# Lookup Dictionaries
Certifi = list(df['certifications'].unique()); certi_code = list(df['certifications_code'].unique()); C = dict(zip(Certifi,certi_code))
Workshops = list(df['workshops'].unique()); Workshops_code = list(df['workshops_code'].unique()); W = dict(zip(Workshops,Workshops_code))
Interested_subjects = list(df['Interested subjects'].unique()); Interested_subjects_code = list(df['Interested subjects_code'].unique()); ISC = dict(zip(Interested_subjects,Interested_subjects_code))
interested_career_area = list(df['interested career area '].unique()); interested_career_area_code = list(df['interested career area _code'].unique()); ICA = dict(zip(interested_career_area,interested_career_area_code))
Typeofcompany = list(df['Type of company want to settle in?'].unique()); Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique()); TOCO = dict(zip(Typeofcompany,Typeofcompany_code))
Interested_Books = list(df['Interested Type of Books'].unique()); Interested_Books_code = list(df['Interested Type of Books_code'].unique()); IB = dict(zip(Interested_Books,Interested_Books_code))

FEATURE_NAMES = [
    "Logical quotient rating", "coding skills rating", "hackathons", "public speaking points", "self-learning capability?",
    "Extra-courses did", "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert", "reading_and_writing_skills",
    "memory_capability_score", "B_hard worker", "B_smart worker", "A_Management", "A_Technical", "Interested subjects_code",
    "Interested Type of Books_code", "certifications_code", "workshops_code", "Type of company want to settle in?_code", "interested_career_area _code"
]

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie URLs for some visual flair
lottie_career = load_lottieurl("https://lottie.host/9e8b7c5b-3b0b-4b1a-9c1a-1a2b3c4d5e6f/6z7x8c9v0b.json") # Fallback to a random if this fails
lottie_auth = load_lottieurl("https://lottie.host/818f95f4-307a-4b9d-8c1d-1e5b1e6e9e6c/7q8w9e0r1t.json")

# --- CUSTOM CSS ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        * { font-family: 'Outfit', sans-serif; }
        
        .main {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: white;
        }
        
        div.stButton > button:first-child {
            background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 30px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(106, 17, 203, 0.4);
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
        }
        
        h1, h2, h3 {
            color: #ffffff;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        
        .stTextInput > div > div > input {
            background-color: rgba(255,255,255,0.05);
            color: white;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Validation Functions
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_phone(phone):
    return re.match(r"^\+?1?\d{9,15}$", phone)

def inputlist(
    Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points,
    self_learning_capability, Extra_courses_did, Taken_inputs_from_seniors_or_elders,
    worked_in_teams_ever, Introvert, reading_and_writing_skills, memory_capability_score,
    smart_or_hard_work, Management_or_Techinical, Interested_subjects, 
    Interested_Type_of_Books, certifications, workshops, 
    Type_of_company_want_to_settle_in, interested_career_area
):
    features = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]
    
    def yes_no(v): return 1 if v == "Yes" else 0
    features.extend([yes_no(self_learning_capability), yes_no(Extra_courses_did), 
                     yes_no(Taken_inputs_from_seniors_or_elders), yes_no(worked_in_teams_ever), yes_no(Introvert)])

    range_map = {"poor": 0, "medium": 1, "excellent": 2}
    features.append(range_map[reading_and_writing_skills])
    features.append(range_map[memory_capability_score])

    is_hard = 1 if smart_or_hard_work in ("Hard worker", "Hard Worker") else 0
    is_smart = 1 if smart_or_hard_work in ("Smart worker", "smart worker") else 0
    features.extend([is_hard, is_smart])

    is_management = 1 if Management_or_Techinical == "Management" else 0
    is_technical = 1 if Management_or_Techinical == "Technical" else 0
    features.extend([is_management, is_technical])

    features.append(ISC[Interested_subjects])
    features.append(IB[Interested_Type_of_Books])
    features.append(C[certifications])
    features.append(W[workshops])
    features.append(TOCO[Type_of_company_want_to_settle_in])
    features.append(ICA[interested_career_area])

    probs = regressor.predict_proba([features])[0]
    classes = regressor.classes_
    label = classes[int(np.argmax(probs))]
    return label, probs, classes

def main():
    st.set_page_config(page_title="Careermore AI", page_icon="👨🏻‍💻", layout="wide")
    local_css()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        col1, col2 = st.columns([1, 1])
        with col1:
            if lottie_auth:
                st_lottie(lottie_auth, height=400, key="auth")
            else:
                st.image("https://img.freepik.com/free-vector/authentication-concept-illustration_114360-2168.jpg", use_container_width=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.title("Welcome to Careermore")
            menu = ["Login", "Register"]
            choice = st.selectbox("Menu", menu)

            if choice == "Login":
                username = st.text_input("Email/Username")
                password = st.text_input("Password", type='password')
                if st.button("Login"):
                    if login_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")
            else:
                new_user = st.text_input("Choose Email/Username")
                new_password = st.text_input("Choose Password", type='password')
                if st.button("Register"):
                    if validate_email(new_user):
                        if add_user(new_user, new_password):
                            st.success("Account created! Please Login.")
                        else:
                            st.warning("Username already exists.")
                    else:
                        st.error("Please enter a valid email for username.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # --- LOGGED IN UI ---
        
        # TOP CORNER PROFILE SECTION
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        t1, t2 = st.columns([5, 1.5]) 
        with t2:
            st.markdown(f"""
                <div style="background: rgba(106, 17, 203, 0.2); backdrop-filter: blur(10px); padding: 12px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                    <p style="margin:0; font-size: 0.75rem; color: #b191ff; text-transform: uppercase; letter-spacing: 1px;">Session Active</p>
                    <p style="margin:0; font-weight: 600; font-size: 0.9rem; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{st.session_state.username}">{st.session_state.username}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🚪 Sign Out", key="logout_btn_top", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()

        # SIDEBAR NAVIGATION
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Menu", ["🚀 Prediction", "👤 User Profile", "📊 Analytics"])
        
        st.markdown('<div style="text-align:center;"><h1>👨🏻‍💻 Careermore AI Predictor 👨🏻‍💻</h1></div>', unsafe_allow_html=True)
        
        if page == "🚀 Prediction":
            col1, col2 = st.columns([1, 2])
            with col1:
                if lottie_career:
                    st_lottie(lottie_career, height=300)
                else:
                    st.image("https://img.freepik.com/free-vector/choice-worker-concept-illustration_114360-5026.jpg", use_container_width=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Personal Details")
                name = st.text_input("Full Name")
                contact = st.text_input("Contact Number")
                email = st.text_input("Email ID", value=st.session_state.username)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Skill Ratings & Preferences")
                c1, c2 = st.columns(2)
                with c1:
                    lq = st.slider('Logical Quotient', 0, 10, 5)
                    cs = st.slider('Coding Skills', 0, 10, 5)
                    hacks = st.slider('Hackathons', 0, 10, 0)
                    ps = st.slider('Public Speaking', 0, 10, 5)
                with c2:
                    slc = st.selectbox('Self Learning', ('Yes', 'No'))
                    ext = st.selectbox('Extra Courses', ('Yes', 'No'))
                    advice = st.selectbox('Took Advice', ('Yes', 'No'))
                    team = st.selectbox('Team Work', ('Yes', 'No'))
                
                c3, c4 = st.columns(2)
                with c3:
                    intro = st.selectbox('Introvert', ('Yes', 'No'))
                    rw = st.selectbox('Reading/Writing', ('poor', 'medium', 'excellent'))
                    mem = st.selectbox('Memory Score', ('poor', 'medium', 'excellent'))
                    work_type = st.selectbox('Work Habit', ('Smart worker', 'Hard Worker'))
                with c4:
                    stream = st.selectbox('Stream', ('Management', 'Technical'))
                    subj = st.selectbox('Interests', ('programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking'))
                    books = st.selectbox('Favorite Books', ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'))
                    certs = st.selectbox('Certifications', ('information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python'))
                
                c5, c6 = st.columns(2)
                with c5:
                    worksh = st.selectbox('Workshops', ('Testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies'))
                with c6:
                    comp = st.selectbox('Company Type', ('BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based'))
                
                area = st.selectbox('Career Area', ('testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing'))
                st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Predict Your Career 🚀"):
                # VALIDATION
                if not name:
                    st.error("Please enter your name.")
                elif not validate_phone(contact):
                    st.error("Please enter a valid phone number.")
                elif not validate_email(email):
                    st.error("Please enter a valid email.")
                else:
                    with st.spinner("Analyzing your profile..."):
                        label, probs, classes = inputlist(
                            lq, cs, hacks, ps, slc, ext, advice, team, intro, rw, mem, 
                            work_type, stream, subj, books, certs, worksh, comp, area
                        )
                        time.sleep(1.5)
                        st.balloons()
                        st.success(f"### Predicted Career: {label}")
                        
                        top_indices = np.argsort(probs)[::-1][:3]
                        st.info("#### Top 3 Recommendations")
                        for idx in top_indices:
                            st.write(f"🎯 **{classes[idx]}** — Confidence: {probs[idx]:.2%}")
                        
                        add_data(name, contact, email, lq, cs, hacks, ps, slc, ext, advice, team, intro, rw, mem, work_type, stream, subj, books, certs, worksh, comp, area, label)

        elif page == "👤 User Profile":
            st.header("Your History")
            rows = fetch_all_by_user(st.session_state.username)
            if rows:
                cols_names = ["Name", "Phone", "Email", "LQ", "CS", "Hacks", "PS", "SLC", "Ext", "Advice", "Team", "Intro", "RW", "Mem", "Work", "Stream", "Subj", "Books", "Certs", "Worksh", "Comp", "Area", "Prediction"]
                df_user = pd.DataFrame(rows, columns=cols_names)
                st.dataframe(df_user)
            else:
                st.write("No predictions yet.")

        elif page == "📊 Analytics":
            st.header("Community Insights")
            role_counts = fetch_role_counts()
            if role_counts:
                role_df = pd.DataFrame(role_counts, columns=["Role", "Count"])
                st.bar_chart(role_df.set_index("Role"))
            else:
                st.info("No data available yet.")

if __name__ == '__main__':
    main()
