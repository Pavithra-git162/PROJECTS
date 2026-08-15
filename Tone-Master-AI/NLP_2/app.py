import streamlit as st
import google.generativeai as genai
import re
import sqlite3
import hashlib
import hmac
import os
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
        salt TEXT NOT NULL,
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

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000).hex()

def verify_password(password, salt_hex, hash_hex):
    salt = bytes.fromhex(salt_hex)
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, hash_hex)

def register_user(con, username, password):
    if len(username.strip()) == 0:
        return False, "Username cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = os.urandom(16)
    pwd_hash = hash_password(password, salt)
    try:
        con.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?,?,?,?)",
            (username.strip().lower(), pwd_hash, salt.hex(), datetime.now().isoformat()),
        )
        con.commit()
        return True, "Account created — please log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."

def login_user(con, username, password):
    row = con.execute(
        "SELECT id, password_hash, salt FROM users WHERE username=?",
        (username.strip().lower(),),
    ).fetchone()
    if not row:
        return False, "Username not found.", None
    uid, pwd_hash, salt_hex = row
    if not verify_password(password, salt_hex, pwd_hash):
        return False, "Incorrect password.", None
    return True, f"Welcome back, {username.strip()}!", uid

def save_email(con, user_id, original, rewritten, tone, language, similarity):
    con.execute(
        """INSERT INTO email_history
        (user_id, original_text, rewritten_text, tone, language, similarity_score, created_at)
        VALUES (?,?,?,?,?,?,?)""",
        (user_id, original, rewritten, tone, language, similarity, datetime.now().isoformat()),
    )
    con.commit()

def load_history(con, user_id):
    return con.execute(
        """SELECT id, original_text, rewritten_text, tone, language, similarity_score, created_at
        FROM email_history WHERE user_id=? ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()

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
        for name in [
            "gemini-2.5-flash", "models/gemini-2.5-flash",
            "gemini-2.5-flash-preview-05-20", "models/gemini-2.5-flash-preview-05-20",
        ]:
            try:
                m = genai.GenerativeModel(name)
                m.generate_content("Hello")
                return m, True, name
            except Exception:
                continue
        return None, False, None
    except Exception:
        return None, False, None

def sanitize_input(text):
    replacements = {
        r"\bmy bad\b": "I apologize", r"\byo\b": "Hello", r"\bhey boss\b": "Dear Supervisor",
        r"\bsorry about this\b": "I apologize for any inconvenience",
        r"\bhonestly ive been super busy\b": "I have been occupied with other tasks",
        r"\bother stuff\b": "other responsibilities", r"\btotally forgot\b": "I overlooked",
        r"\bill have it ready\b": "I will prepare it", r"\bfor sure\b": "certainly",
        r"\bim facing\b": "I am encountering", r"\bill try to fix it\b": "I will attempt to resolve it",
        r"\bsorry\b": "I apologize",
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
TONE_DESCS = list(TONE_INSTRUCTIONS.keys())

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
        if "quota" in msg or "limit" in msg or "resource" in msg:
            return "QUOTA_ERROR"
        if "safety" in msg or "blocked" in msg:
            return "SAFETY_ERROR"
        return "API_ERROR"

def compute_similarity(t1, t2):
    try:
        v = TfidfVectorizer()
        m = v.fit_transform([t1, t2])
        return round(float(cosine_similarity(m[0], m[1])[0][0]) * 100, 1)
    except Exception:
        return 0.0

def similarity_label(s):
    if s >= 75:
        return "Very similar to original", "amber"
    if s >= 50:
        return "Moderately changed", "blue"
    if s >= 25:
        return "Significantly rewritten", "green"
    return "Completely transformed", "violet"

COLOR_HEX = {
    "amber": "#B45309", "blue": "#1D4ED8", "green": "#15803D", "violet": "#6D28D9",
}

# ════════════════════════════════════════════════
# PAGE CONFIG + THEME
# ════════════════════════════════════════════════
st.set_page_config(page_title="Tone Master AI", page_icon="✉️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #FAFAFC; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1080px; }

/* Sidebar: minimal dark accent, no fragile input overrides */
section[data-testid="stSidebar"] {
    background: #1A1730;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #E7E5F5 !important;
}

/* Buttons — one consistent style, no internal-class targeting */
div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}
div.stButton > button[kind="primary"] {
    background: #4F46E5;
    border: none;
}
div.stButton > button[kind="primary"]:hover {
    background: #4338CA;
}

/* Cards */
.tm-card {
    background: #FFFFFF;
    border: 1px solid #E7E5EF;
    border-radius: 14px;
    padding: 22px 26px;
    box-shadow: 0 1px 3px rgba(20,20,43,0.04);
}

/* Hero */
.tm-hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: #14142B;
    margin: 0 0 0.35rem 0;
}
.tm-hero h1 span { color: #4F46E5; }
.tm-hero p {
    font-size: 15px;
    color: #55516B;
    margin: 0;
}

.tm-label {
    font-size: 11px;
    font-weight: 700;
    color: #6B6685;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.1rem 0 0.5rem 0;
    display: block;
}

/* Output card */
.oc-badges { display: flex; gap: 8px; margin-bottom: 14px; }
.oc-badge {
    font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 20px;
}
.oc-badge.tone { background: #EDEBFF; color: #4F46E5; }
.oc-badge.lang { background: #F0F0F5; color: #43405B; }
.oc-text { font-size: 15px; line-height: 1.8; color: #14142B; white-space: pre-wrap; }

/* Metric strip */
.m-strip { display: flex; gap: 12px; margin-top: 1rem; }
.m-box {
    flex: 1; background: #FFFFFF; border: 1px solid #E7E5EF; border-radius: 12px;
    padding: 14px 16px; text-align: center;
}
.m-lbl { font-size: 10.5px; font-weight: 700; color: #8B87A0; text-transform: uppercase; letter-spacing: 0.07em; display: block; margin-bottom: 4px; }
.m-val { font-family: 'Fraunces', serif; font-size: 24px; font-weight: 600; color: #14142B; display: block; }
.m-sub { font-size: 11px; color: #8B87A0; font-weight: 500; display: block; margin-top: 2px; }
.sim-track { background: #EEEDF6; border-radius: 4px; height: 5px; margin-top: 8px; overflow: hidden; }
.sim-fill { height: 100%; border-radius: 4px; }

/* Notice */
.tm-notice {
    background: #EDEBFF; border: 1px solid #C7C2F5; border-radius: 10px;
    padding: 12px 16px; font-size: 13.5px; color: #34306B; margin-top: 0.75rem;
}

/* History cards */
.h-card {
    background: #FFFFFF; border: 1px solid #E7E5EF; border-radius: 14px;
    padding: 16px 20px; margin-bottom: 10px;
}
.h-meta { font-size: 12px; color: #6B6685; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.h-orig {
    font-size: 13px; color: #43405B; font-style: italic; line-height: 1.6;
    background: #F7F6FC; border-left: 3px solid #C7C2F5; border-radius: 0 8px 8px 0;
    padding: 8px 12px; margin-bottom: 10px;
}
.h-rewritten { font-size: 14px; color: #14142B; line-height: 1.75; white-space: pre-wrap; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.badge.tone { background: #4F46E5; color: #FFFFFF; }
.badge.lang { background: #F0F0F5; color: #43405B; }

.page-header { padding: 4px 0 18px 0; }
.page-header h2 { font-family: 'Fraunces', serif; font-weight: 600; color: #14142B; margin: 0; }
.page-header p { color: #6B6685; font-size: 14px; margin: 4px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════
if "db" not in st.session_state:
    st.session_state.db = init_db()
con = st.session_state.db

defaults = {
    "logged_in": False, "user_id": None, "username": "",
    "page": "rewrite", "selected_tone": "Professional",
    "example": "", "rewritten_email": None,
    "sim_score": None, "last_input": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

try:
    if "GEMINI_API_KEY" in st.secrets and "api_key" not in st.session_state:
        st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if "model" not in st.session_state:
    if "api_key" in st.session_state:
        m, ok, _ = configure_gemini(st.session_state.api_key)
        st.session_state.model = m if ok else None

# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style='padding:0.4rem 0 0.6rem 0;'>
            <div style='font-family:"Fraunces",serif;font-size:1.35rem;color:#FFFFFF;font-weight:600;'>
                ✉️ Tone Master AI
            </div>
            <div style='font-size:11px;color:#9E9AC2;margin-top:2px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;'>
                AI-Powered Email Rewriter
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    if not st.session_state.logged_in:
        mode = st.radio("Account", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")

        if mode == "Login":
            st.caption("Sign in to your account")
            lu = st.text_input("Username", key="sb_lu", placeholder="Enter your username")
            lp = st.text_input("Password", type="password", key="sb_lp", placeholder="Enter your password")
            if st.button("Login", key="btn_login", type="primary", use_container_width=True):
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
            st.caption("Create an account")
            nu = st.text_input("Choose a username", key="sb_nu", placeholder="Must be unique")
            np_val = st.text_input("Choose a password", type="password", key="sb_np", placeholder="At least 6 characters")
            if st.button("Create Account", key="btn_signup", type="primary", use_container_width=True):
                if nu and np_val:
                    ok, msg = register_user(con, nu, np_val)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in both fields.")

    else:
        st.markdown(
            f"""
            <div style='margin-bottom:0.8rem;'>
                <div style='font-size:10.5px;color:#9E9AC2;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:3px;'>Signed in as</div>
                <div style='font-size:16px;font-weight:600;color:#FFFFFF;'>{st.session_state.username}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav = st.radio(
            "Navigate",
            ["Rewrite Email", "Email History"],
            index=0 if st.session_state.page == "rewrite" else 1,
            label_visibility="collapsed",
        )
        st.session_state.page = "rewrite" if nav == "Rewrite Email" else "history"

        st.divider()
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
    st.markdown(
        """
        <div class='page-header'>
            <h2>Email History</h2>
            <p>Everything you've rewritten, saved to your account.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.logged_in:
        st.markdown("<div class='tm-notice'>Please log in to view your saved email history.</div>", unsafe_allow_html=True)
        st.stop()

    rows = load_history(con, st.session_state.user_id)

    if not rows:
        st.markdown(
            """
            <div style='text-align:center;padding:3rem 0;'>
                <div style='font-family:"Fraunces",serif;font-size:1.4rem;color:#14142B;margin-bottom:6px;'>No history yet</div>
                <div style='font-size:14px;color:#6B6685;'>Rewrite your first email and it will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        tf = st.selectbox("Filter by Tone", ["All Tones"] + TONE_DESCS, key="hf_tone")
    with fc2:
        lf = st.selectbox("Filter by Language", ["All Languages", "English", "Tamil", "Hindi"], key="hf_lang")
    with fc3:
        sq = st.text_input("Search emails", placeholder="Type a keyword to search...", key="hf_search")

    visible = [
        r for r in rows
        if (tf == "All Tones" or r[3] == tf)
        and (lf == "All Languages" or r[4] == lf)
        and (not sq or sq.lower() in r[1].lower() or sq.lower() in r[2].lower())
    ]

    st.caption(f"Showing {len(visible)} of {len(rows)} saved emails")

    for row in visible:
        eid, orig, rewritten, tone, lang, sim, created_at = row
        date_str = created_at[:10]
        sim_lbl, sim_key = similarity_label(sim) if sim else ("No score", "blue")
        sim_color = COLOR_HEX.get(sim_key, "#43405B")

        st.markdown(
            f"""
            <div class='h-card'>
                <div class='h-meta'>
                    <span>{date_str}</span>
                    <span class='badge tone'>{tone}</span>
                    <span class='badge lang'>{lang}</span>
                    <span style='color:{sim_color};font-weight:700;'>{sim}% · {sim_lbl}</span>
                </div>
                <div class='h-orig'>{orig[:200]}{"..." if len(orig) > 200 else ""}</div>
                <div class='h-rewritten'>{rewritten}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ba, bb, _ = st.columns([1.3, 1, 4])
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

st.markdown(
    """
    <div class='tm-hero'>
        <h1>Rewrite any email, <span>any tone.</span></h1>
        <p>Paste your draft. Choose a tone and language. Get a polished email in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<span class='tm-label'>Try an example</span>", unsafe_allow_html=True)
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("Project update with a bug", use_container_width=True):
        st.session_state.example = (
            "Hey! So I finally got the homepage redesign done today. "
            "Looks pretty good but there's this annoying bug where the mobile menu "
            "keeps overlapping on smaller screens. I'll try to fix it tomorrow. Let me know!"
        )
        st.rerun()
with ex2:
    if st.button("Missed deadline excuse", use_container_width=True):
        st.session_state.example = (
            "My bad, I know the report was due today but honestly I've been super busy "
            "with other stuff and totally forgot about it. Can I get it to you by Friday? "
            "Sorry about this, won't happen again!"
        )
        st.rerun()
with ex3:
    if st.button("Meeting follow-up", use_container_width=True):
        st.session_state.example = (
            "Yo! Just wanted to follow up on our meeting yesterday. "
            "I think we should go with option B because it's cheaper and faster. "
            "Also, can you send me those files you mentioned? Thanks!"
        )
        st.rerun()

st.markdown("<span class='tm-label'>Select a tone</span>", unsafe_allow_html=True)
tcols = st.columns(5)
for i, tone in enumerate(TONE_DESCS):
    with tcols[i]:
        is_sel = st.session_state.selected_tone == tone
        if st.button(tone, key=f"t_{tone}", use_container_width=True,
                     type="primary" if is_sel else "secondary"):
            st.session_state.selected_tone = tone
            st.rerun()

left, right = st.columns([3, 1])
with left:
    st.markdown("<span class='tm-label'>Your email draft</span>", unsafe_allow_html=True)
    email_input = st.text_area(
        "Email draft", height=200,
        value=st.session_state.get("example", ""),
        placeholder="Paste your rough email here",
        label_visibility="collapsed",
    )
with right:
    st.markdown("<span class='tm-label'>Language</span>", unsafe_allow_html=True)
    language_choice = st.selectbox(
        "Language", ["English", "Tamil", "Hindi"],
        label_visibility="collapsed", key="lang_sel",
    )
    st.markdown(
        f"""
        <div class='tm-card' style='margin-top:10px;padding:14px 16px;'>
            <span class='tm-label' style='margin:0 0 4px 0;'>Active tone</span>
            <div style='font-size:16px;font-weight:600;color:#14142B;'>{st.session_state.selected_tone}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.button("✦ Rewrite Email", type="primary", use_container_width=True):
    if not email_input.strip():
        st.warning("Please enter your email text before rewriting.")
        st.stop()

    cleaned = sanitize_input(email_input)
    with st.spinner(f"Rewriting in {st.session_state.selected_tone} tone..."):
        result = gemini_rewrite(st.session_state.model, cleaned, st.session_state.selected_tone, language_choice)

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

if st.session_state.rewritten_email:
    result = st.session_state.rewritten_email
    sim = st.session_state.sim_score or 0.0
    sim_lbl, sim_key = similarity_label(sim)
    sim_color = COLOR_HEX.get(sim_key, "#43405B")
    orig_input = st.session_state.last_input or email_input

    st.markdown(
        f"""
        <div class='tm-card' style='margin-top:1.2rem;'>
            <div class='oc-badges'>
                <span class='oc-badge tone'>{st.session_state.selected_tone}</span>
                <span class='oc-badge lang'>{language_choice}</span>
            </div>
            <div class='oc-text'>{result}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    orig_wc = len(orig_input.split()) if orig_input.strip() else 0
    new_wc = len(result.split())
    diff = new_wc - orig_wc
    diff_color = "#15803D" if diff <= 0 else "#B45309"

    st.markdown(
        f"""
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
                <span class='m-lbl'>Word change</span>
                <span class='m-val' style='color:{diff_color};'>{'+' if diff > 0 else ''}{diff}</span>
                <span class='m-sub'>vs your draft</span>
            </div>
            <div class='m-box'>
                <span class='m-lbl'>Similarity</span>
                <span class='m-val' style='color:{sim_color};'>{sim}%</span>
                <span class='m-sub' style='color:{sim_color};font-weight:600;'>{sim_lbl}</span>
                <div class='sim-track'><div class='sim-fill' style='width:{sim}%;background:{sim_color};'></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇ Download Rewritten Email (.txt)",
        data=result,
        file_name=f"email_{st.session_state.selected_tone.lower()}_{language_choice.lower()}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    if not st.session_state.logged_in:
        st.markdown(
            "<div class='tm-notice'>Login to automatically save this email to your history and access it anytime.</div>",
            unsafe_allow_html=True,
        )
