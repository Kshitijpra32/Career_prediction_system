import sqlite3
import threading
import bcrypt

conn = sqlite3.connect("CBRSdata.db", check_same_thread=False)
db_lock = threading.Lock()

def create_table():
    with db_lock:
        # User authentication table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users("
            "username TEXT UNIQUE,"
            "password TEXT"
            ")"
        )

        # Prediction table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictiontable("
            "Name TEXT,"
            "Contact_Number INTEGER,"
            "Email_address TEXT,"
            "Logical_quotient_rating INTEGER,"
            "coding_skills_rating INTEGER,"
            "hackathons INTEGER,"
            "public_speaking_points INTEGER,"
            "self_learning_capability TEXT,"
            "Extra_courses_did TEXT,"
            "Taken_inputs_from_seniors_or_elders TEXT,"
            "worked_in_teams_ever TEXT,"
            "Introvert TEXT,"
            "reading_and_writing_skills TEXT,"
            "memory_capability_score TEXT,"
            "smart_or_hard_work TEXT,"
            "Management_or_Techinical TEXT,"
            "Interested_subjects TEXT,"
            "Interested_Type_of_Books TEXT,"
            "certifications TEXT,"
            "workshops TEXT,"
            "Type_of_company_want_to_settle_in TEXT,"
            "interested_career_area TEXT,"
            "Predicted_role TEXT"
            ")"
        )
        conn.commit()

def add_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    with db_lock:
        try:
            conn.execute("INSERT INTO users(username, password) VALUES (?,?)", (username, hashed_pw))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def login_user(username, password):
    with db_lock:
        data = conn.execute("SELECT password FROM users WHERE username=?", (username,)).fetchone()
    if data:
        return bcrypt.checkpw(password.encode('utf-8'), data[0])
    return False

def add_data(
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
    Predicted_role,
):
    with db_lock:
        conn.execute(
            "INSERT INTO predictiontable("
            "Name,Contact_Number,Email_address,"
            "Logical_quotient_rating,coding_skills_rating,hackathons,public_speaking_points,"
            "self_learning_capability,Extra_courses_did,Taken_inputs_from_seniors_or_elders,"
            "worked_in_teams_ever,Introvert,reading_and_writing_skills,memory_capability_score,"
            "smart_or_hard_work,Management_or_Techinical,Interested_subjects,Interested_Type_of_Books,"
            "certifications,workshops,Type_of_company_want_to_settle_in,interested_career_area,"
            "Predicted_role"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
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
                Predicted_role,
            ),
        )
        conn.commit()

def fetch_all_by_user(email):
    with db_lock:
        return conn.execute("SELECT * FROM predictiontable WHERE Email_address=?", (email,)).fetchall()

def fetch_role_counts():
    with db_lock:
        return conn.execute(
            "SELECT Predicted_role, COUNT(*) FROM predictiontable "
            "WHERE Predicted_role IS NOT NULL "
            "GROUP BY Predicted_role"
        ).fetchall()
