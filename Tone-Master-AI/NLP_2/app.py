import streamlit as st
import google.generativeai as genai
import re
import sqlite3
import hashlib
import hmac
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════
def init_db():
    con = sqlite3.connect("tonemaster.db", check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS email_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        original_text TEXT NOT NULL,
        rewritten_text TEXT NOT NULL,
        tone TEXT NOT NULL,
        language TEXT NOT NULL,
        similarity_score REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id))""")
    con.commit()
    return con

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
def verify_password(p, h): return hmac.compare_digest(hash_password(p), h)

def register_user(con, username, password):
    if len(username.strip()) == 0: return False, "Username cannot be empty."
    if len(password) < 6: return False, "Password must be at least 6 characters."
    try:
        con.execute("INSERT INTO users (username,password_hash,created_at) VALUES (?,?,?)",
                    (username.strip().lower(), hash_password(password), datetime.now().isoformat()))
        con.commit(); return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def login_user(con, username, password):
    row = con.execute("SELECT id,password_hash FROM users WHERE username=?",
                      (username.strip().lower(),)).fetchone()
    if not row: return False, "Username not found.", None
    if not verify_password(password, row[1]): return False, "Incorrect password.", None
    return True, f"Welcome back, {username.strip()}!", row[0]

def save_email(con, user_id, original, rewritten, tone, language, similarity):
    con.execute("""INSERT INTO email_history
        (user_id,original_text,rewritten_text,tone,language,similarity_score,created_at)
        VALUES (?,?,?,?,?,?,?)""",
        (user_id, original, rewritten, tone, language, similarity, datetime.now().isoformat()))
    con.commit()

def load_history(con, user_id):
    return con.execute("""SELECT id,original_text,rewritten_text,tone,language,
        similarity_score,created_at FROM email_history
        WHERE user_id=? ORDER BY created_at DESC""", (user_id,)).fetchall()

def delete_email(con, email_id, user_id):
    con.execute("DELETE FROM email_history WHERE id=? AND user_id=?", (email_id, user_id))
    con.commit()

# ════════════════════════════════════════════════
# GEMINI
# ════════════════════════════════════════════════
@st.cache_resource
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        for name in ["gemini-2.5-flash","models/gemini-2.5-flash",
                     "gemini-2.5-flash-preview-05-20","models/gemini-2.5-flash-preview-05-20"]:
            try:
                m = genai.GenerativeModel(name)
                m.generate_content("Hello")
                return m, True, name
            except: continue
        return None, False, None
    except: return None, False, None

def sanitize_input(text):
    replacements = {
        r'\bmy bad\b':'I apologize', r'\byo\b':'Hello', r'\bhey boss\b':'Dear Supervisor',
        r'\bsorry about this\b':'I apologize for any inconvenience',
        r'\bhonestly ive been super busy\b':'I have been occupied with other tasks',
        r'\bother stuff\b':'other responsibilities', r'\btotally forgot\b':'I overlooked',
        r'\bill have it ready\b':'I will prepare it', r'\bfor sure\b':'certainly',
        r'\bim facing\b':'I am encountering', r'\bill try to fix it\b':'I will attempt to resolve it',
        r'\bsorry\b':'I apologize',
    }
    for p, r in replacements.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

TONE_INSTRUCTIONS = {
    "Professional": (
        "Rewrite this email in a professional yet natural tone. Sound polished and workplace-appropriate. "
        "Use clear, direct language. Keep output 50-100 words. "
        "FORMAT: 'Subject: [line]' newline 'Dear [Name],' newline body newline 'Best regards,' newline '[Your Name]'."
    ),
    "Casual": (
        "Rewrite this email in a casual, friendly, conversational tone like talking to a close colleague. "
        "Keep output 50-100 words. "
        "FORMAT: 'Subject: [line]' newline 'Hi [Name],' newline body newline 'Thanks,' or 'Cheers,' newline '[Your Name]'."
    ),
    "Concise": (
        "Rewrite this email as briefly and directly as possible. 45-70 words max. "
        "Short punchy sentences. Cut all fluff. "
        "FORMAT: 'Subject: [line]' newline 'Hi [Name],' newline 1 paragraph max newline 'Thanks,' newline '[Your Name]'."
    ),
    "Apologetic": (
        "Rewrite this email with a sincere, professional apology tone. Acknowledge the mistake clearly, "
        "take responsibility without over-explaining, and offer a clear path forward. Keep output 50-100 words. "
        "FORMAT: 'Subject: [line]' newline 'Dear [Name],' newline acknowledge mistake and resolution newline 'Sincerely,' newline '[Your Name]'."
    ),
    "Follow-up": (
        "Rewrite this email as a polite, confident follow-up. Not pushy, not desperate. "
        "Reference the previous communication, restate the key point, end with a clear call to action. "
        "Keep output 50-100 words. "
        "FORMAT: 'Subject: [line]' newline 'Hi [Name],' newline brief context and ask newline 'Looking forward to hearing from you,' newline '[Your Name]'."
    ),
}

TONE_DESCS = ["Professional", "Casual", "Concise", "Apologetic", "Follow-up"]

def gemini_rewrite(model, text, tone, language):
    lang_note = "" if language == "English" else f"Write the ENTIRE email in {language} only. "
    prompt = (
        f"{lang_note}{TONE_INSTRUCTIONS[tone]} "
        "Write only the email. No explanations or commentary.\n\n"
        f"Original email:\n{text}"
    )
    try:
        r = model.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or "limit" in msg or "resource" in msg: return "QUOTA_ERROR"
        if "safety" in msg or "blocked" in msg: return "SAFETY_ERROR"
        return "API_ERROR"

def compute_similarity(t1, t2):
    try:
        v = TfidfVectorizer()
        m = v.fit_transform([t1, t2])
        return round(float(cosine_similarity(m[0], m[1])[0][0]) * 100, 1)
    except: return 0.0

def similarity_label(s):
    if s >= 75: return "Very similar to original", "#0D1E32"
    if s >= 50: return "Moderately changed", "#1B3557"
    if s >= 25: return "Significantly rewritten", "#2E5A8A"
    return "Completely transformed", "#5C7FA6"

# ════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════
st.set_page_config(page_title="Tone Master AI", page_icon="✉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&display=swap');

:root {
    --navy: #1B3557;
    --navy-dark: #0D1E32;
    --navy-mid: #2E5A8A;
    --navy-soft: #5C7FA6;
    --navy-tint: #EAF0F8;
    --navy-tint-2: #DCE6F2;
    --navy-border: #B8CCE0;
    --white: #FFFFFF;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--navy-tint); }
#MainMenu, footer{ visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1100px; }


/* SIDEBAR */
section[data-testid="stSidebar"] { background: var(--navy) !important; }
section[data-testid="stSidebar"] * { color: var(--white) !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--navy-soft) !important;
    color: var(--navy-dark) !important;
    border-radius: 7px !important;
    font-size: 14px !important;
    padding: 6px 10px !important;
    caret-color: var(--navy) !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--white) !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.2) !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #A8C0D8 !important; font-size: 13px !important;
}
section[data-testid="stSidebar"] .stTextInput label {
    font-size: 13px !important; font-weight: 600 !important;
    color: var(--white) !important; margin-bottom: 3px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important; font-weight: 600 !important; color: var(--white) !important;
}

/* Sidebar buttons: navy sidebar bg, so buttons need to be white to stand out,
   with navy text — kept consistent whatever state (hover/active/focus) */
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] .stButton>button p,
section[data-testid="stSidebar"] .stButton>button div,
section[data-testid="stSidebar"] .stButton>button span {
    color: var(--navy-dark) !important;
}
section[data-testid="stSidebar"] .stButton>button {
    background: var(--white) !important;
    border: 1.5px solid var(--white) !important; font-weight: 600 !important;
    border-radius: 7px !important; width: 100% !important;
    padding: 7px 14px !important; font-size: 14px !important; margin-top: 3px !important;
    min-height: 0 !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: var(--navy-tint) !important; border-color: var(--navy-tint) !important;
}
section[data-testid="stSidebar"] .stButton>button:active,
section[data-testid="stSidebar"] .stButton>button:focus,
section[data-testid="stSidebar"] .stButton>button:focus:not(:active) {
    background: var(--navy-tint) !important;
    box-shadow: none !important; outline: none !important;
    border-color: var(--navy-tint) !important;
}
section[data-testid="stSidebar"] .stButton>button:active *,
section[data-testid="stSidebar"] .stButton>button:focus *,
section[data-testid="stSidebar"] .stButton>button:hover * {
    color: var(--navy-dark) !important;
}

/* MAIN BUTTONS — force text color on every possible inner element and state,
   so nothing ever renders blank/white-on-white when clicked or focused */
.stButton>button,
.stButton>button p,
.stButton>button div,
.stButton>button span {
    color: var(--navy-dark) !important;
}
.stButton>button {
    background: var(--white) !important;
    border: 1.5px solid var(--navy-border) !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 500 !important; transition: all 0.15s !important;
    padding: 8px 14px !important; min-height: 0 !important;
}
.stButton>button:hover,
.stButton>button:hover p,
.stButton>button:hover div,
.stButton>button:hover span {
    border-color: var(--navy) !important; background: var(--navy-tint) !important;
    color: var(--navy-dark) !important;
}
.stButton>button:active,
.stButton>button:focus,
.stButton>button:focus:not(:active),
.stButton>button:active p, .stButton>button:active div, .stButton>button:active span,
.stButton>button:focus p, .stButton>button:focus div, .stButton>button:focus span {
    background: var(--navy-tint-2) !important;
    color: var(--navy-dark) !important;
    border-color: var(--navy) !important;
    box-shadow: none !important; outline: none !important;
}

/* Primary (navy) buttons — e.g. Rewrite Email, selected tone */
.stButton>button[kind="primary"],
button[data-testid="stBaseButton-primary"],
.stButton>button[kind="primary"] p,
.stButton>button[kind="primary"] div,
.stButton>button[kind="primary"] span {
    background: var(--navy) !important; color: var(--white) !important;
    border: none !important; font-size: 14px !important; font-weight: 600 !important;
    padding: 8px 14px !important; border-radius: 8px !important; min-height: 0 !important;
}

.stButton>button[kind="primary"]:hover,
.stButton>button[kind="primary"]:hover p,
.stButton>button[kind="primary"]:hover div,
.stButton>button[kind="primary"]:hover span {
    background: var(--navy-mid) !important; color: var(--white) !important;
}
.stButton>button[kind="primary"]:active,
.stButton>button[kind="primary"]:focus,
.stButton>button[kind="primary"]:focus:not(:active),
.stButton>button[kind="primary"]:active p, .stButton>button[kind="primary"]:active div, .stButton>button[kind="primary"]:active span,
.stButton>button[kind="primary"]:focus p, .stButton>button[kind="primary"]:focus div, .stButton>button[kind="primary"]:focus span {
    background: var(--navy-mid) !important;
    color: var(--white) !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Secondary (unselected) buttons — lock background AND text color so nothing
   goes blank when tapped/focused, e.g. the tone selector buttons */
.stButton>button[kind="secondary"],
.stButton>button:not([kind="primary"]),
.stButton>button[kind="secondary"] p, .stButton>button[kind="secondary"] div, .stButton>button[kind="secondary"] span {
    background: var(--white) !important;
    color: var(--navy-dark) !important;
}
.stButton>button[kind="secondary"]:hover,
.stButton>button:not([kind="primary"]):hover {
    border-color: var(--navy) !important;
    background: var(--navy-tint) !important;
    color: var(--navy-dark) !important;
}
.stButton>button[kind="secondary"]:active,
.stButton>button[kind="secondary"]:focus,
.stButton>button[kind="secondary"]:focus:not(:active),
.stButton>button:not([kind="primary"]):active,
.stButton>button:not([kind="primary"]):focus,
.stButton>button:not([kind="primary"]):focus:not(:active) {
    background: var(--navy-tint-2) !important;
    color: var(--navy-dark) !important;
    border-color: var(--navy) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* TEXT AREA */
.stTextArea textarea {
    background: var(--white) !important; border: 1.5px solid var(--navy-border) !important;
    border-radius: 10px !important; font-size: 15px !important;
    color: var(--navy-dark) !important; padding: 12px !important; line-height: 1.7 !important;
    caret-color: var(--navy) !important;
}
.stTextArea textarea:focus {
    border-color: var(--navy) !important;
    box-shadow: 0 0 0 3px rgba(27,53,87,0.15) !important;
}
.stTextArea textarea::placeholder { color: #6E88A4 !important; font-size: 14px !important; }

/* SELECTBOX — Streamlit re-injects this widget's own CSS after ours loads,
   so a plain !important can still lose a specificity tie and render text
   invisible. We repeat the class selector to force higher specificity, and
   also reset -webkit-text-fill-color, which is what actually makes text
   disappear even when "color" looks correct. */
.stSelectbox.stSelectbox > div > div,
.stSelectbox.stSelectbox div[data-baseweb="select"] > div {
    background: var(--white) !important;
    border: 1.5px solid var(--navy-border) !important;
    border-radius: 8px !important;
    min-height: 38px !important;
}
.stSelectbox.stSelectbox div[data-baseweb="select"],
.stSelectbox.stSelectbox div[data-baseweb="select"] *,
.stSelectbox.stSelectbox div[data-baseweb="select"] div,
.stSelectbox.stSelectbox div[data-baseweb="select"] span,
.stSelectbox.stSelectbox div[data-baseweb="select"] p {
    color: var(--navy-dark) !important;
    -webkit-text-fill-color: var(--navy-dark) !important;
    background-color: transparent !important;
    opacity: 1 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.stSelectbox.stSelectbox svg { fill: var(--navy) !important; }

/* The dropdown list Streamlit pops open is rendered outside .stSelectbox,
   appended straight to <body>, so it needs its own top-level rules */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: var(--white) !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li *,
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] li * {
    color: var(--navy-dark) !important;
    -webkit-text-fill-color: var(--navy-dark) !important;
    background: var(--white) !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"],
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background: var(--navy-tint) !important;
    color: var(--navy-dark) !important;
}

/* LABELS */
.sec-label {
    font-size: 11px; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 8px; display: block;
}

/* HERO */
.tm-hero { padding: 0.2rem 0 1rem 0; }
.tm-hero h1 {
    font-family: 'Lora', serif; font-size: 2.2rem; color: var(--navy-dark);
    line-height: 1.2; margin: 0 0 0.4rem 0; font-weight: 400;
}
.tm-hero h1 em { font-style: italic; color: var(--navy); }
.tm-hero p { font-size: 15px; color: var(--navy-mid); margin: 0; line-height: 1.55; }

/* TONE BAR */
.tone-bar {
    height: 3px; background: var(--navy); border-radius: 2px;
    margin-top: -4px; margin-bottom: 2px;
}

/* ACTIVE TONE INFO CARD */
.tone-info-card {
    background: var(--navy-tint); border: 1.5px solid var(--navy-border);
    border-radius: 10px; padding: 10px 12px;
}
.tone-info-card .tic-label {
    font-size: 11px; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 5px; display: block;
}
.tone-info-card .tic-name { font-size: 16px; font-weight: 600; color: var(--navy-dark); }

/* OUTPUT CARD */
.out-card {
    background: var(--white); border: 1.5px solid var(--navy-border);
    border-radius: 12px; padding: 18px; margin-top: 1rem;
}
.out-card .oc-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1.5px solid var(--navy-tint-2);
}
.oc-tone-badge {
    background: var(--navy); color: var(--white); font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
.oc-lang-badge {
    background: var(--white); color: var(--navy); font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; border: 1.5px solid var(--navy);
}
.out-card .oc-text {
    font-size: 15px; color: var(--navy-dark); line-height: 1.8;
    white-space: pre-wrap; font-weight: 400;
}

/* METRICS */
.m-strip { display: flex; gap: 10px; margin-top: 1rem; }
.m-box {
    background: var(--white); border: 1.5px solid var(--navy-border);
    border-radius: 10px; padding: 10px 12px; flex: 1; text-align: center;
}
.m-box .m-lbl {
    font-size: 11px; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 5px; display: block;
}
.m-box .m-val {
    font-family: 'Lora', serif; font-size: 22px; font-weight: 500;
    color: var(--navy-dark); margin-bottom: 2px; display: block;
}
.m-box .m-sub { font-size: 11px; color: var(--navy-mid); font-weight: 500; display: block; }
.sim-track { background: var(--navy-tint-2); border-radius: 4px; height: 5px; margin-top: 8px; overflow: hidden; }
.sim-fill { height: 100%; border-radius: 4px; }

/* INFO NOTICE */
.info-notice {
    background: var(--navy-tint); border: 1.5px solid var(--navy-border); border-radius: 8px;
    padding: 10px 14px; font-size: 14px; color: var(--navy-dark);
    margin-top: 0.75rem; font-weight: 500; line-height: 1.5;
}

/* PAGE HEADER (History) */
.page-header {
    background: var(--navy); border-radius: 12px;
    padding: 20px 26px; margin-bottom: 1.25rem;
}
.page-header h2 {
    font-family: 'Lora', serif; color: var(--white);
    font-size: 1.7rem; font-weight: 400; margin: 0 0 4px 0;
}
.page-header p { color: #C8DCF0; font-size: 14px; margin: 0; }

/* HISTORY CARDS */
.h-card {
    background: var(--white); border: 1.5px solid var(--navy-border);
    border-radius: 12px; padding: 14px 18px;
    margin-bottom: 10px; transition: border-color 0.15s;
}
.h-card:hover { border-color: var(--navy); }
.h-meta {
    font-size: 12px; font-weight: 700; color: var(--navy);
    letter-spacing: 0.07em; text-transform: uppercase;
    margin-bottom: 10px; display: flex;
    align-items: center; flex-wrap: wrap; gap: 6px;
}
.h-orig {
    font-size: 13px; color: var(--navy-dark); margin-bottom: 10px;
    border-left: 3px solid var(--navy); padding: 8px 12px;
    font-style: italic; line-height: 1.6;
    background: var(--navy-tint); border-radius: 0 6px 6px 0;
}
.h-rewritten { font-size: 14px; color: var(--navy-dark); line-height: 1.75; white-space: pre-wrap; }
.badge-tone {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 700; background: var(--navy); color: var(--white);
}
.badge-lang {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 600; background: var(--white);
    color: var(--navy); border: 1.5px solid var(--navy);
}
section[data-testid="stSidebar"] .stTextInput button {
    display: none !important;
}         
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# INIT SESSION STATE
# ════════════════════════════════════════════════
if "db" not in st.session_state:
    st.session_state.db = init_db()
con = st.session_state.db

defaults = {
    "logged_in": False, "user_id": None, "username": "",
    "page": "rewrite", "selected_tone": "Professional",
    "email_draft_input": "", "rewritten_email": None,
    "sim_score": None, "last_input": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

try:
    if "GEMINI_API_KEY" in st.secrets and "api_key" not in st.session_state:
        st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
except: pass

if "model" not in st.session_state:
    if "api_key" in st.session_state:
        m, ok, _ = configure_gemini(st.session_state.api_key)
        st.session_state.model = m if ok else None

# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1rem 0;'>
        <div style='font-family:"Lora",serif;font-size:1.4rem;color:#FFFFFF;font-weight:400;'>
           ✉ Tone Master AI
        </div>
        <div style='font-size:11px;color:#C8DCF0;margin-top:4px;letter-spacing:.08em;
                    text-transform:uppercase;font-weight:600;'>
            AI-Powered Email Rewriter
        </div>
    </div>
    <hr style='border-color:#2E5A8A;margin:0 0 1rem 0;'>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        mode = st.radio("", ["Login", "Sign Up"], horizontal=True)

        if mode == "Login":
            st.markdown("<div style='font-size:15px;font-weight:700;color:#FFFFFF;margin-bottom:10px;'>Sign in to your account</div>", unsafe_allow_html=True)
            lu = st.text_input("Username", key="sb_lu", placeholder="Enter your username")

            lp = st.text_input(
                "Password",
                type="password",
                key="sb_lp",
                placeholder="Enter your password"
            )

            if st.button("Login", key="btn_login"):
                if lu and lp:
                    ok, msg, uid = login_user(con, lu, lp)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_id = uid
                        st.session_state.username = lu.strip()
                        st.session_state.page = "rewrite"
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in both fields.")
        else:
            st.markdown("<div style='font-size:15px;font-weight:700;color:#FFFFFF;margin-bottom:10px;'>Create an account</div>", unsafe_allow_html=True)
            nu = st.text_input("Choose a username", key="sb_nu", placeholder="Must be unique")
            np_val = st.text_input("Choose a password", type="password", key="sb_np", placeholder="At least 6 characters")
            if st.button("Create Account", key="btn_signup"):
                if nu and np_val:
                    ok, msg = register_user(con, nu, np_val)
                    if ok: st.success(msg + " Please log in.")
                    else: st.error(msg)
                else:
                    st.warning("Please fill in both fields.")

        st.markdown("<hr style='border-color:#2E5A8A;margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px;color:#C8DCF0;line-height:1.6;'></div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style='margin-bottom:1rem;'>
            <div style='font-size:11px;color:#C8DCF0;text-transform:uppercase;
                        letter-spacing:.09em;font-weight:700;margin-bottom:4px;'>Signed in as</div>
            <div style='font-size:17px;font-weight:600;color:#FFFFFF;'>{st.session_state.username}</div>
        </div>
        """, unsafe_allow_html=True)

        on_rewrite = st.session_state.page == "rewrite"
        on_history = st.session_state.page == "history"

        if st.button(
            f"{'▶ ' if on_rewrite else ''}Rewrite Email",
            key="nav_rewrite",
            use_container_width=True
        ):
            st.session_state.page = "rewrite"
            st.rerun()

        if st.button(
            f"{'▶ ' if on_history else ''}Email History",
            key="nav_history",
            use_container_width=True
        ):
            st.session_state.page = "history"
            st.rerun()

        st.markdown("<hr style='border-color:#2E5A8A;margin:1rem 0;'>", unsafe_allow_html=True)
        if st.button("Logout", key="btn_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = ""
            st.session_state.page = "rewrite"
            st.session_state.rewritten_email = None
            st.session_state.sim_score = None
            st.session_state.last_input = ""
            st.rerun()

# ════════════════════════════════════════════════
# PAGE: HISTORY
# ════════════════════════════════════════════════
if st.session_state.page == "history":
    st.markdown("""
    <div class='page-header'>
        <h2>Email History</h2>
        <p></p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.markdown("<div class='info-notice'>Please log in to view your saved email history.</div>", unsafe_allow_html=True)
        st.stop()

    rows = load_history(con, st.session_state.user_id)

    if not rows:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;'>
            <div style='font-family:"Lora",serif;font-size:1.5rem;color:#0D1E32;margin-bottom:8px;'>No history yet</div>
            <div style='font-size:14px;color:#2A4A6A;'>Rewrite your first email and it will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        tf = st.selectbox("Filter by Tone", ["All Tones"] + list(TONE_INSTRUCTIONS.keys()), key="hf_tone")
    with fc2:
        lf = st.selectbox("Filter by Language", ["All Languages","English","Tamil","Hindi"], key="hf_lang")
    with fc3:
        sq = st.text_input("Search emails", placeholder="Type a keyword to search...", key="hf_search")

    visible = [r for r in rows
               if (tf == "All Tones" or r[3] == tf)
               and (lf == "All Languages" or r[4] == lf)
               and (not sq or sq.lower() in r[1].lower() or sq.lower() in r[2].lower())]

    st.markdown(f"<div style='font-size:13px;color:#1B3557;margin:0.75rem 0 1rem 0;font-weight:600;'>Showing {len(visible)} of {len(rows)} saved emails</div>", unsafe_allow_html=True)

    for row in visible:
        eid, orig, rewritten, tone, lang, sim, created_at = row
        date_str = created_at[:10]
        sim_lbl, sim_color = similarity_label(sim) if sim else ("No score", "#2A4A6A")

        st.markdown(f"""
        <div class='h-card'>
            <div class='h-meta'>
                <span>{date_str}</span>&nbsp;&middot;&nbsp;
                <span class='badge-tone'>{tone}</span>
                <span class='badge-lang'>{lang}</span>&nbsp;&middot;&nbsp;
                <span class='sim-chip' style='background:{sim_color}18;color:{sim_color};border:1.5px solid {sim_color}60;'>
                    {sim}% — {sim_lbl}
                </span>
            </div>
            <div class='h-orig'>{orig[:200]}{"..." if len(orig) > 200 else ""}</div>
            <div class='h-rewritten'>{rewritten}</div>
        </div>
        """, unsafe_allow_html=True)

        ba, bb, _ = st.columns([1.2, 1, 4])
        with ba:
            if st.button("Reuse this original", key=f"reuse_{eid}", use_container_width=True):
                st.session_state.email_draft_input = orig
                st.session_state.page = "rewrite"
                st.rerun()
        with bb:
            if st.button("Delete", key=f"del_{eid}", use_container_width=True):
                delete_email(con, eid, st.session_state.user_id)
                st.rerun()

    st.stop()

# ════════════════════════════════════════════════
# PAGE: REWRITE
# ════════════════════════════════════════════════
if "api_key" not in st.session_state or not st.session_state.api_key:
    st.error("API key not configured. Please add GEMINI_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()
if not st.session_state.get("model"):
    st.error("Could not connect to Gemini. Please check your API key and internet connection.")
    st.stop()

# Hero
st.markdown("""
<div class='tm-hero'>
    <h1>Rewrite any email, <em>any tone.</em></h1>
    <p style="font-family:'Lora', serif; font-weight:500;">Paste your draft. Choose a tone and language. Get a polished email in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Example buttons
st.markdown("<span class='sec-label'>TRY AN EXAMPLE</span>", unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("Project update with a bug", use_container_width=True):
        st.session_state.email_draft_input = (
            "Hey! So I finally got the homepage redesign done today. "
            "Looks pretty good but there's this annoying bug where the mobile menu "
            "keeps overlapping on smaller screens. I'll try to fix it tomorrow. Let me know!")
        st.rerun()
with ex2:
    if st.button("Missed deadline excuse", use_container_width=True):
        st.session_state.email_draft_input = (
            "My bad, I know the report was due today but honestly I've been super busy "
            "with other stuff and totally forgot about it. Can I get it to you by Friday? "
            "Sorry about this, won't happen again!")
        st.rerun()
with ex3:
    if st.button("Meeting follow-up", use_container_width=True):
        st.session_state.email_draft_input = (
            "Yo! Just wanted to follow up on our meeting yesterday. "
            "I think we should go with option B because it's cheaper and faster. "
            "Also, can you send me those files you mentioned? Thanks!")
        st.rerun()

# Tone selector
st.markdown("<span class='sec-label' style='margin-top:0.75rem;display:block;'>SELECT A TONE</span>", unsafe_allow_html=True)
tcols = st.columns(5)
for i, tone in enumerate(TONE_DESCS):
    with tcols[i]:
        is_sel = st.session_state.selected_tone == tone
        if st.button(tone, key=f"t_{tone}", use_container_width=True,
                     type="primary" if is_sel else "secondary"):
            st.session_state.selected_tone = tone
            st.rerun()
        if is_sel:
            st.markdown("<div class='tone-bar'></div>", unsafe_allow_html=True)

# Draft input + language
left, right = st.columns([3, 1])
with left:
    st.markdown("<span class='sec-label' style='margin-top:0.5rem;display:block;'>YOUR EMAIL DRAFT</span>", unsafe_allow_html=True)
    email_input = st.text_area(
        "Email draft",
        height=200,
        key="email_draft_input",
        placeholder="Paste your rough email here",
        label_visibility="collapsed"
    )
with right:
    st.markdown("<span class='sec-label' style='margin-top:0.5rem;display:block;'>CHOOSE LANGUAGE</span>", unsafe_allow_html=True)
    language_choice = st.selectbox(
        "Language", ["English", "Tamil", "Hindi"],
        label_visibility="collapsed", key="lang_sel"
    )
    sel = st.session_state.selected_tone
    st.markdown(f"""
    <div class='tone-info-card' style='margin-top:8px;'>
        <span class='tic-label'>Active Tone</span>
        <div class='tic-name'>{sel}</div>
    </div>
    """, unsafe_allow_html=True)

# Rewrite button
if st.button("✦ Rewrite Email", type="primary", use_container_width=True):
    if not email_input.strip():
        st.warning("Please enter your email text before rewriting.")
        st.stop()

    cleaned = sanitize_input(email_input)
    with st.spinner(f"Rewriting in {st.session_state.selected_tone} tone..."):
        result = gemini_rewrite(st.session_state.model, cleaned,
                                st.session_state.selected_tone, language_choice)

    if result == "QUOTA_ERROR":
        st.error("API quota exceeded. Please wait a minute and try again.")
        st.stop()
    if result == "SAFETY_ERROR":
        st.error("Content was flagged by safety filters. Please rephrase and try again.")
        st.stop()
    if result == "API_ERROR":
        st.error("Unable to process. Please check your internet connection and try again.")
        st.stop()

    sim = compute_similarity(email_input, result)
    st.session_state.rewritten_email = result
    st.session_state.sim_score = sim
    st.session_state.last_input = email_input

    if st.session_state.logged_in:
        save_email(con, st.session_state.user_id, email_input, result,
                   st.session_state.selected_tone, language_choice, sim)

# Output
if st.session_state.rewritten_email:
    result     = st.session_state.rewritten_email
    sim        = st.session_state.sim_score or 0.0
    sim_lbl, sim_color = similarity_label(sim)
    orig_input = st.session_state.last_input or email_input

    st.markdown(f"""
    <div class='out-card'>
        <div class='oc-header'>
            <span class='oc-tone-badge'>{st.session_state.selected_tone}</span>
            <span class='oc-lang-badge'>{language_choice}</span>
        </div>
        <div class='oc-text'>{result}</div>
    </div>
    """, unsafe_allow_html=True)

    orig_wc = len(orig_input.split()) if orig_input.strip() else 0
    new_wc  = len(result.split())
    diff    = new_wc - orig_wc
    diff_color = "#0D1E32" if diff <= 0 else "#2E5A8A"

    st.markdown(f"""
    <div class='m-strip'>
        <div class='m-box'>
            <span class='m-lbl'>Original</span>
            <span class='m-val'>{orig_wc}</span>
            <span class='m-sub'>words in draft</span>
        </div>
        <div class='m-box'>
            <span class='m-lbl'>Rewritten</span>
            <span class='m-val'>{new_wc}</span>
            <span class='m-sub'>words in output</span>
        </div>
        <div class='m-box'>
            <span class='m-lbl'>Word Change</span>
            <span class='m-val' style='color:{diff_color};'>{'+' if diff > 0 else ''}{diff}</span>
            <span class='m-sub'>vs your draft</span>
        </div>
        <div class='m-box'>
            <span class='m-lbl'>Similarity</span>
            <span class='m-val' style='color:{sim_color};'>{sim}%</span>
            <span class='m-sub' style='color:{sim_color};font-weight:600;'>{sim_lbl}</span>
            <div class='sim-track'>
                <div class='sim-fill' style='width:{sim}%;background:{sim_color};'></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "⬇ Download Rewritten Email (.txt)",
        data=result,
        file_name=f"email_{st.session_state.selected_tone.lower()}_{language_choice.lower()}.txt",
        mime="text/plain",
        use_container_width=True
    )

    if not st.session_state.logged_in:
        st.markdown("""
        <div class='info-notice'>
             Login to automatically save this email to your history and access it anytime.
        </div>
        """, unsafe_allow_html=True)
