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
    if s >= 75: return "Very similar to original", "#92400E"
    if s >= 50: return "Moderately changed", "#1E40AF"
    if s >= 25: return "Significantly rewritten", "#14532D"
    return "Completely transformed", "#4C1D95"

# ════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════
st.set_page_config(page_title="Tone Master AI", page_icon="✉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F0F4FA; }
#MainMenu, footer{ visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1100px; }


/* SIDEBAR */
section[data-testid="stSidebar"] { background: #1B3557 !important; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #FFFFFF !important;
    border: 2px solid #4A90C4 !important;
    color: #0D1E32 !important;
    border-radius: 7px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    caret-color: #1B3557 !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.2) !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #7A9AB8 !important; font-size: 13px !important;
}
section[data-testid="stSidebar"] .stTextInput label {
    font-size: 13px !important; font-weight: 600 !important;
    color: #C8DFF2 !important; margin-bottom: 3px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important; font-weight: 600 !important; color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton>button {
    background: #2E7DC4 !important; color: #FFFFFF !important;
    border: none !important; font-weight: 600 !important;
    border-radius: 7px !important; width: 100% !important;
    padding: 10px 16px !important; font-size: 14px !important; margin-top: 3px !important;
}
section[data-testid="stSidebar"] .stButton>button:hover { background: #1A6AAE !important; }

/* Active nav button highlight */
section[data-testid="stSidebar"] .stButton>button.nav-active {
    background: #FFFFFF !important; color: #1B3557 !important;
}

/* MAIN BUTTONS */
.stButton>button {
    background: #FFFFFF !important; color: #0D1E32 !important;
    border: 2px solid #B8CCE0 !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 500 !important; transition: all 0.15s !important;
}
.stButton>button:hover {
    border-color: #1B3557 !important; background: #E8F0FA !important;
}
.stButton>button[kind="primary"] {
    background: #1B3557 !important; color: #FFFFFF !important;
    border: none !important; font-size: 15px !important; font-weight: 600 !important;
    padding: 12px 0 !important; border-radius: 8px !important;
}
.stButton>button[kind="primary"]:hover { background: #254A7A !important; }

/* TEXT AREA */
.stTextArea textarea {
    background: #FFFFFF !important; border: 2px solid #B8CCE0 !important;
    border-radius: 10px !important; font-size: 15px !important;
    color: #0D1E32 !important; padding: 12px !important; line-height: 1.7 !important;
    caret-color: #1B3557 !important;
}
.stTextArea textarea:focus {
    border-color: #1B3557 !important;
    box-shadow: 0 0 0 3px rgba(27,53,87,0.1) !important;
}
.stTextArea textarea::placeholder { color: #7A96B0 !important; font-size: 14px !important; }

/* SELECTBOX */
.stSelectbox > div > div {
    background: #FFFFFF !important; border: 2px solid #B8CCE0 !important;
    border-radius: 8px !important; color: #0D1E32 !important;
    font-size: 14px !important; font-weight: 500 !important;
}

/* LABELS */
.sec-label {
    font-size: 11px; font-weight: 700; color: #1B3557;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 8px; display: block;
}

/* HERO */
.tm-hero { padding: 0.2rem 0 1rem 0; }
.tm-hero h1 {
    font-family: 'Lora', serif; font-size: 2.2rem; color: #0D1E32;
    line-height: 1.2; margin: 0 0 0.4rem 0; font-weight: 400;
}
.tm-hero h1 em { font-style: italic; color: #1E6FAA; }
.tm-hero p { font-size: 15px; color: #2A4A6A; margin: 0; line-height: 1.55; }

/* TONE BAR */
.tone-bar {
    height: 3px; background: #1B3557; border-radius: 2px;
    margin-top: -4px; margin-bottom: 2px;
}

/* ACTIVE TONE INFO CARD */
.tone-info-card {
    background: #DAE8F5; border: 2px solid #5A9AD0;
    border-radius: 10px; padding: 12px;
}
.tone-info-card .tic-label {
    font-size: 11px; font-weight: 700; color: #1B3557;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 5px; display: block;
}
.tone-info-card .tic-name { font-size: 16px; font-weight: 600; color: #0D1E32; }

/* OUTPUT CARD */
.out-card {
    background: #FFFFFF; border: 2px solid #B8CCE0;
    border-radius: 12px; padding: 20px; margin-top: 1rem;
}
.out-card .oc-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1.5px solid #D8E8F4;
}
.oc-tone-badge {
    background: #1B3557; color: #FFFFFF; font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
.oc-lang-badge {
    background: #DAE8F5; color: #0D3A6A; font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; border: 1.5px solid #5A9AD0;
}
.out-card .oc-text {
    font-size: 15px; color: #0D1E32; line-height: 1.8;
    white-space: pre-wrap; font-weight: 400;
}

/* METRICS */
.m-strip { display: flex; gap: 10px; margin-top: 1rem; }
.m-box {
    background: #FFFFFF; border: 2px solid #B8CCE0;
    border-radius: 10px; padding: 12px 14px; flex: 1; text-align: center;
}
.m-box .m-lbl {
    font-size: 11px; font-weight: 700; color: #1B3557;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin-bottom: 5px; display: block;
}
.m-box .m-val {
    font-family: 'Lora', serif; font-size: 22px; font-weight: 500;
    color: #0D1E32; margin-bottom: 2px; display: block;
}
.m-box .m-sub { font-size: 11px; color: #2A4A6A; font-weight: 500; display: block; }
.sim-track { background: #C8DCF0; border-radius: 4px; height: 5px; margin-top: 8px; overflow: hidden; }
.sim-fill { height: 100%; border-radius: 4px; }

/* INFO NOTICE */
.info-notice {
    background: #DAE8F5; border: 2px solid #5A9AD0; border-radius: 8px;
    padding: 12px 16px; font-size: 14px; color: #0D3055;
    margin-top: 0.75rem; font-weight: 500; line-height: 1.5;
}

/* PAGE HEADER (History) */
.page-header {
    background: #1B3557; border-radius: 12px;
    padding: 20px 26px; margin-bottom: 1.25rem;
}
.page-header h2 {
    font-family: 'Lora', serif; color: #FFFFFF;
    font-size: 1.7rem; font-weight: 400; margin: 0 0 4px 0;
}
.page-header p { color: #A8C8E8; font-size: 14px; margin: 0; }

/* HISTORY CARDS */
.h-card {
    background: #FFFFFF; border: 2px solid #B8CCE0;
    border-radius: 12px; padding: 16px 20px;
    margin-bottom: 10px; transition: border-color 0.15s;
}
.h-card:hover { border-color: #1B3557; }
.h-meta {
    font-size: 12px; font-weight: 700; color: #1B3557;
    letter-spacing: 0.07em; text-transform: uppercase;
    margin-bottom: 10px; display: flex;
    align-items: center; flex-wrap: wrap; gap: 6px;
}
.h-orig {
    font-size: 13px; color: #1B3A5A; margin-bottom: 10px;
    border-left: 3px solid #5A9AD0; padding: 8px 12px;
    font-style: italic; line-height: 1.6;
    background: #EEF5FC; border-radius: 0 6px 6px 0;
}
.h-rewritten { font-size: 14px; color: #0D1E32; line-height: 1.75; white-space: pre-wrap; }
.badge-tone {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 700; background: #1B3557; color: #FFFFFF;
}
.badge-lang {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 600; background: #DAE8F5;
    color: #0D3055; border: 1.5px solid #5A9AD0;
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
    "example": "", "rewritten_email": None,
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
        <div style='font-size:11px;color:#A8C8E8;margin-top:4px;letter-spacing:.08em;
                    text-transform:uppercase;font-weight:600;'>
            AI-Powered Email Rewriter
        </div>
    </div>
    <hr style='border-color:#2E5A8A;margin:0 0 1rem 0;'>
    """, unsafe_allow_html=True)

    if not st.session_state.logged_in:
        mode = st.radio("", ["Login", "Sign Up"], horizontal=True)

        if mode == "Login":
            st.markdown("<div style='font-size:13px;color:#C8DFF2;margin-bottom:10px;font-weight:500;'><h4>Sign in to your account</h4></div>", unsafe_allow_html=True)
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
            st.markdown("<div style='font-size:13px;color:#C8DFF2;margin-bottom:10px;font-weight:500;'><h4>Create an account</h4></div>", unsafe_allow_html=True)
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
        st.markdown("<div style='font-size:12px;color:#A8C8E8;line-height:1.6;'></div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style='margin-bottom:1rem;'>
            <div style='font-size:11px;color:#A8C8E8;text-transform:uppercase;
                        letter-spacing:.09em;font-weight:700;margin-bottom:4px;'>Signed in as</div>
            <div style='font-size:17px;font-weight:600;color:#FFFFFF;'>{st.session_state.username}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── FIX: use query_params to drive navigation reliably ──
        # Highlight active page button
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
                st.session_state.example = orig
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
    <p><h4 style="font-family:'Playfair Display', serif;  font-weight:500;">Paste your draft. Choose a tone and language. Get a polished email in seconds.</h4></p>
</div>
""", unsafe_allow_html=True)

# Example buttons
st.markdown("<span class='sec-label'><h5>TRY AN EXAMPLE</h5></span>", unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("Project update with a bug", use_container_width=True):
        st.session_state.example = (
            "Hey! So I finally got the homepage redesign done today. "
            "Looks pretty good but there's this annoying bug where the mobile menu "
            "keeps overlapping on smaller screens. I'll try to fix it tomorrow. Let me know!")
        st.rerun()
with ex2:
    if st.button("Missed deadline excuse", use_container_width=True):
        st.session_state.example = (
            "My bad, I know the report was due today but honestly I've been super busy "
            "with other stuff and totally forgot about it. Can I get it to you by Friday? "
            "Sorry about this, won't happen again!")
        st.rerun()
with ex3:
    if st.button("Meeting follow-up", use_container_width=True):
        st.session_state.example = (
            "Yo! Just wanted to follow up on our meeting yesterday. "
            "I think we should go with option B because it's cheaper and faster. "
            "Also, can you send me those files you mentioned? Thanks!")
        st.rerun()

# Tone selector
st.markdown("<span class='sec-label' style='margin-top:0.75rem;display:block;'><h5>SELECT A TONE</h5></span>", unsafe_allow_html=True)
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
    st.markdown("<span class='sec-label' style='margin-top:0.5rem;display:block;'><h5>YOUR EMAIL DRAFT</h5></span>", unsafe_allow_html=True)
    email_input = st.text_area(
        "Email draft",
        height=200,
        value=st.session_state.get("example", ""),
        placeholder="Paste your rough email here",
        label_visibility="collapsed"
    )
with right:
    st.markdown("<span class='sec-label' style='margin-top:0.5rem;display:block;'><h6>CHOOSE LANGUAGE</h6></span>", unsafe_allow_html=True)
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
    diff_color = "#14532D" if diff <= 0 else "#92400E"

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
