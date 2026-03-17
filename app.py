# **1. Importing Necessary Libraries** 📚

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import streamlit as st
from db import *

pickleFile=open("weights.pkl","rb")
regressor=pickle.load(pickleFile) # our model

# **2. Loading Dataset**

df = pd.read_csv('./data/mldata.csv')
df.head()

df['workshops'] = df['workshops'].replace(['testing'],'Testing')
df.head()

print(df.columns.unique)

n = df['Suggested Job Role'].unique()
print(len(n))

print('The shape of our training set: %s professionals and %s features'%(df.shape[0],df.shape[1]))


# **5. Feature Engineering**

## (a) Binary Encoding for Categorical Variables

newdf = df
newdf.head(10)

cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    print(i)
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (b) Number Encoding for Categorical 

mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    print(i)    
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())

## (c) Dummy Variable Encoding

print(df['Management or Technical'].unique())
print(df['hard/smart worker'].unique())

df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
df.head()

df.sort_values(by=['certifications'])

print("List of Numerical features: \n" , df.select_dtypes(include=np.number).columns.tolist())


category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']]
for i in category_cols:
  print(i)

Certifi = list(df['certifications'].unique())
print(Certifi)
certi_code = list(df['certifications_code'].unique())
print(certi_code)

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)

Certi_l = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certi_l,certi_code))

Workshops = list(df['workshops'].unique())
print(Workshops)
Workshops_code = list(df['workshops_code'].unique())
print(Workshops_code)
W = dict(zip(Workshops,Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
print(Interested_subjects)
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects,Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
print(interested_career_area)
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area,interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
print(Typeofcompany)
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany,Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
print(Interested_subjects)
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books,Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}
print(Range_dict)


A = 'yes'
B = 'No'
col = [A,B]
for i in col:
  if(i=='yes'):
    i = 1
  print(i)


f =[]
A = 'r programming'
clms = ['r programming',0]
for i in clms:
  for key in C:
    if(i==key):
      i = C[key]
      f.append(i)
print(f)

C = dict(zip(Certifi,certi_code))
  
print(C)

array = np.array([1, 2, 3, 4])
array.reshape(-1, 1)


FEATURE_NAMES = [
    "Logical quotient rating",
    "coding skills rating",
    "hackathons",
    "public speaking points",
    "self-learning capability?",
    "Extra-courses did",
    "Taken inputs from seniors or elders",
    "worked in teams ever?",
    "Introvert",
    "reading and writing skills",
    "memory capability score",
    "B_hard worker",
    "B_smart worker",
    "A_Management",
    "A_Technical",
    "Interested subjects_code",
    "Interested Type of Books_code",
    "certifications_code",
    "workshops_code",
    "Type of company want to settle in?_code",
    "interested career area _code",
]


def inputlist(
    Name,
    Contact_Number,
    Email_address,
    Logical_quotient_rating,
    coding_skills_rating,
    hackathons,
    public_speaking_points,
    self_learning_capability,
    Extra_courses_did,
    Taken_inputs_from_seniors_or_elders,
    worked_in_teams_ever,
    Introvert,
    reading_and_writing_skills,
    memory_capability_score,
    smart_or_hard_work,
    Management_or_Techinical,
    Interested_subjects,
    Interested_Type_of_Books,
    certifications,
    workshops,
    Type_of_company_want_to_settle_in,
    interested_career_area,
):
  # 1) Raw numeric ratings
  features = [
      Logical_quotient_rating,
      coding_skills_rating,
      hackathons,
      public_speaking_points,
  ]

  # 2) Yes/No binary fields
  def yes_no(v: str) -> int:
    return 1 if v == "Yes" else 0

  features.extend(
      [
          yes_no(self_learning_capability),
          yes_no(Extra_courses_did),
          yes_no(Taken_inputs_from_seniors_or_elders),
          yes_no(worked_in_teams_ever),
          yes_no(Introvert),
      ]
  )

  # 3) Ordinal skills
  range_map = {"poor": 0, "medium": 1, "excellent": 2}
  features.append(range_map[reading_and_writing_skills])
  features.append(range_map[memory_capability_score])

  # 4) Dummy variables for hard/smart worker
  is_hard = 1 if smart_or_hard_work in ("Hard worker", "Hard Worker") else 0
  is_smart = 1 if smart_or_hard_work in ("Smart worker", "smart worker") else 0
  features.extend([is_hard, is_smart])

  # 5) Dummy variables for Management or Technical
  is_management = 1 if Management_or_Techinical == "Management" else 0
  is_technical = 1 if Management_or_Techinical == "Technical" else 0
  features.extend([is_management, is_technical])

  # 6) Encoded categorical codes using lookup dicts
  features.append(ISC[Interested_subjects])
  features.append(IB[Interested_Type_of_Books])
  features.append(C[certifications])
  features.append(W[workshops])
  features.append(TOCO[Type_of_company_want_to_settle_in])
  features.append(ICA[interested_career_area])

  if len(features) != 21:
    raise ValueError(f"Unexpected feature vector length {len(features)} (expected 21)")

  probs = regressor.predict_proba([features])[0]
  classes = regressor.classes_
  label = classes[int(np.argmax(probs))]

  return label, probs, classes

def main():

  # with st.spinner('Wait for it...'):
  #     time.sleep(5)
  # st.success('Done!')

  html1="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h1>👨🏻‍💻 Careermore 👨🏻‍💻</h1>
    </div>
      """
  st.markdown(html1,unsafe_allow_html=True) #simple html 

  # Images

  col1, col2, col3 = st.columns(3)

  with col1:
      st.image("./assets/Career _Isometric.png")

  with col2:
      st.image("./assets/career.png")

  with col3:
      st.image("./assets/Career _Outline.png")

  html2="""
    <div style="text-align:center; text-shadow: 3px 1px 2px purple;">
      <h2>Your Friendly Career Advisor<h2>
    </div>
      """
  st.markdown(html2,unsafe_allow_html=True) #simple html 
 
  st.sidebar.title("Your Information")

  Name = st.sidebar.text_input("Full Name")

  Contact_Number = st.sidebar.text_input("Contact Number")

  Email_address = st.sidebar.text_input("Email address")

  if not Name and Email_address:
    st.sidebar.warning("Please fill out your name and EmailID")

  if Name and Contact_Number and Email_address:
    st.sidebar.success("Thanks!")

  Logical_quotient_rating = st.slider(
    'Rate your Logical quotient Skills', 0,10,1)
  st.write(Logical_quotient_rating)

  coding_skills_rating = st.slider(
    'Rate your Coding Skills', 0,10,1)
  st.write(coding_skills_rating)

  hackathons = st.slider(
    'Enter number of Hackathons participated',0,10,1)
  st.write(hackathons)

  public_speaking_points = st.slider(
    'Rate Your Public Speaking', 0,10,1)
  st.write(public_speaking_points)

  self_learning_capability = st.selectbox(
    'Self Learning Capability',
    ('Yes', 'No')
    )
  # st.write('You selected:', self_learning_capability)

  Extra_courses_did = st.selectbox(
    'Extra courses',
  ('Yes', 'No')
  )
  # st.write('You selected:', Extra_courses_did)

  Taken_inputs_from_seniors_or_elders = st.selectbox(
    'Took advice from seniors or elders',
    ('Yes', 'No')
    )
  # st.write('You selected:', Taken_inputs_from_seniors_or_elders)

  worked_in_teams_ever = st.selectbox(
    'Team Co-ordination Skill',
    ('Yes', 'No')
    )
  # st.write('You selected:', worked_in_teams_ever)

  Introvert = st.selectbox(
    'Introvert',
    ('Yes', 'No')
    )
  # st.write('You selected:', Introvert)

  reading_and_writing_skills = st.selectbox(
    'Reading and writing skills',
    ('poor','medium','excellent')
    )
  st.write('You selected: **{}**' .format(reading_and_writing_skills))

  memory_capability_score = st.selectbox(
    'Memory capability score',
    ('poor','medium','excellent')
    )
  st.write('You selected: **{}**' .format(memory_capability_score))

  smart_or_hard_work = st.selectbox(
    'Smart or Hard Work',
    ('Smart worker', 'Hard Worker')
    )
  st.write('You selected: **{}**' .format(smart_or_hard_work))

  Management_or_Techinical = st.selectbox(
    'Management or Techinical',
    ('Management', 'Technical')
    )
  st.write('You selected: **{}**' .format(Management_or_Techinical))

  Interested_subjects = st.selectbox(
    'Interested Subjects',
    ('programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking')
    )
  st.write('You selected: **{}**' .format(Interested_subjects))

  Interested_Type_of_Books = st.selectbox(
    'Interested Books Category',
    ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art', 'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure', 'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery')
    )
  st.write('You selected: **{}**' .format(Interested_Type_of_Books))

  certifications = st.selectbox(
    'Interested_Type_of_Books',
    ('information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack', 'hadoop', 'app development', 'python')
    )
  st.write('You selected: **{}**' .format(certifications))

  workshops = st.selectbox(
    'Workshops Attended',
    ('Testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies')
    )
  st.write('You selected: **{}**' .format(workshops))
  
  Type_of_company_want_to_settle_in = st.selectbox(
    'Type of Company You Want to Settle In ',
    ('BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Product based', 'Service Based')
    )
  st.write('You selected: **{}**' .format(Type_of_company_want_to_settle_in))
  
  interested_career_area = st.selectbox(
    'Interested Career Area',
    ('testing', 'system developer', 'Business process analyst', 'security', 'developer', 'cloud computing')
    )
  st.write('You selected: **{}**' .format(interested_career_area))
  
  result=""
  
  if st.button("Predict"):
    result, probs, classes = inputlist(
        Name,
        Contact_Number,
        Email_address,
        Logical_quotient_rating,
        coding_skills_rating,
        hackathons,
        public_speaking_points,
        self_learning_capability,
        Extra_courses_did,
        Taken_inputs_from_seniors_or_elders,
        worked_in_teams_ever,
        Introvert,
        reading_and_writing_skills,
        memory_capability_score,
        smart_or_hard_work,
        Management_or_Techinical,
        Interested_subjects,
        Interested_Type_of_Books,
        certifications,
        workshops,
        Type_of_company_want_to_settle_in,
        interested_career_area,
    )

    # Progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

    # Balloons
    st.balloons()

    # Main predicted option
    st.success("Predicted Career Option : {}".format(result))

    # Top-3 recommended careers with probabilities
    top_indices = np.argsort(probs)[::-1][:3]
    st.subheader("Top 3 Recommended Career Options")
    for idx in top_indices:
      st.write(
          f"{classes[idx]} — probability {probs[idx]:.2%}"
      )

    # Global feature importance (from the trained tree)
    if hasattr(regressor, "feature_importances_"):
      importances = regressor.feature_importances_
      imp_df = pd.DataFrame(
          {"feature": FEATURE_NAMES, "importance": importances}
      ).sort_values("importance", ascending=False)
      st.subheader("Most Influential Features (overall model)")
      st.bar_chart(imp_df.set_index("feature").head(10))

    # Plot
    corr = df[['Logical quotient rating', 'hackathons', 
           'coding skills rating', 'public speaking points']].corr()
    f,axes = plt.subplots(1,1,figsize = (10,10))
    sns.heatmap(corr,square=True,annot = True,linewidth = .4,center = 2,ax = axes)
    st.subheader("Here are some nerdy analytics 😁")
    st.text("Correlation Between Numerical Features")
    st.pyplot(f)

    # Expander
    with st.expander("See explanation"):
     st.write("""
         The plot above shows the correlation of the features.
         As we can see, no highly correlated pair is found!
     """)

    create_table()
    add_data(
        Name,
        Contact_Number,
        Email_address,
        Logical_quotient_rating,
        coding_skills_rating,
        hackathons,
        public_speaking_points,
        self_learning_capability,
        Extra_courses_did,
        Taken_inputs_from_seniors_or_elders,
        worked_in_teams_ever,
        Introvert,
        reading_and_writing_skills,
        memory_capability_score,
        smart_or_hard_work,
        Management_or_Techinical,
        Interested_subjects,
        Interested_Type_of_Books,
        certifications,
        workshops,
        Type_of_company_want_to_settle_in,
        interested_career_area,
        result,
    )

  # Analytics section: view saved predictions from SQLite
  st.header("Analytics from Saved Predictions")
  if st.button("Load analytics"):
    rows = fetch_all()
    if rows:
      columns = [
          "Name",
          "Contact_Number",
          "Email_address",
          "Logical_quotient_rating",
          "coding_skills_rating",
          "hackathons",
          "public_speaking_points",
          "self_learning_capability",
          "Extra_courses_did",
          "Taken_inputs_from_seniors_or_elders",
          "worked_in_teams_ever",
          "Introvert",
          "reading_and_writing_skills",
          "memory_capability_score",
          "smart_or_hard_work",
          "Management_or_Techinical",
          "Interested_subjects",
          "Interested_Type_of_Books",
          "certifications",
          "workshops",
          "Type_of_company_want_to_settle_in",
          "interested_career_area",
          "Predicted_role",
      ]
      db_df = pd.DataFrame(rows, columns=columns)
      st.subheader("Recent predictions")
      st.dataframe(db_df.tail(25))

      role_counts = fetch_role_counts()
      if role_counts:
        role_df = pd.DataFrame(role_counts, columns=["Predicted_role", "count"])
        st.subheader("Most common predicted careers")
        st.bar_chart(role_df.set_index("Predicted_role"))
    else:
      st.write("No predictions saved yet.")

  # if choice == "Add Post":
  #     st.subheader("Add Your Article")
  #     create_table()
  #     blog_title = st.text_input('Enter Post Title')
  #     blog_author = st.text_input("Enter Author Name",max_chars=50)
  #     blog_article = st.text_area("Enter Your Message",height=200)
  #     blog_post_date = st.date_input("Post Date")
  #     if st.button("Add"):
  #       add_data(blog_author,blog_title,blog_article,blog_post_date)
  #       st.success("Post::'{}' Saved".format(blog_title))

  html3="""

    <div style="color:yellow; margin:80px; text-align:center;">
      Developed with ❤️ by Nikita Mishra</a>
    </div>
      """

  st.markdown(html3,unsafe_allow_html=True)

if __name__=='__main__':
    main()
