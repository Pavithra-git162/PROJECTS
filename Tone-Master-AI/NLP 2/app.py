import streamlit as st
import google.generativeai as genai
import re

# ---------------- GEMINI CONFIG ----------------
@st.cache_resource
def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        # Use only Gemini 2.5-flash models
        model_names = [
            "gemini-2.5-flash",
            "models/gemini-2.5-flash",
            "gemini-2.5-flash-preview-05-20",
            "models/gemini-2.5-flash-preview-05-20",
        ]
        for name in model_names:
            try:
                model = genai.GenerativeModel(name)
                model.generate_content("Hello")
                return model, True, name
            except:
                continue
        return None, False, None
    except:
        return None, False, None

# ---------------- UTIL FUNCTIONS ----------------
def sanitize_input(text):
    replacements = {
        r'\bmy bad\b': 'I apologize',
        r'\byo\b': 'Hello',
        r'\bhey boss\b': 'Dear Supervisor',
        r'\bsorry about this\b': 'I apologize for any inconvenience',
        r'\bhonestly ive been super busy\b': 'I have been occupied with other tasks',
        r'\bother stuff\b': 'other responsibilities',
        r'\btotally forgot\b': 'I overlooked',
        r'\bill have it ready\b': 'I will prepare it',
        r'\bfor sure\b': 'certainly',
        r'\bim facing\b': 'I am encountering',
        r'\bill try to fix it\b': 'I will attempt to resolve it',
        r'\bsorry\b': 'I apologize',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def gemini_rewrite(model, text, tone):
    tone_instructions = {
        "Professional": "Rewrite this email in a professional yet natural tone. Sound polished and workplace-appropriate, but avoid being overly formal or robotic. Use clear, direct language that maintains professionalism while keeping a human touch. Keep the output between 50-100 words regardless of input length - extract and convey only the key points. FORMAT: Start with 'Subject: [subject line]', then greeting 'Dear [Recipient Name],', then 2-3 short paragraphs (only if content needs it), then closing 'Best regards,' and end with '[Your Name]'.",
        
        "Casual": "Rewrite this email in a casual, relaxed, and friendly tone. Make it feel conversational and easy-going, like talking to a close colleague. Use natural, everyday language. Keep the output between 50-100 words regardless of input length - focus on the main message in a friendly way. FORMAT: Start with 'Subject: [subject line]', then greeting 'Hi [Recipient Name],' or 'Hey [Recipient Name],', then 1-2 paragraphs, then closing 'Best,' or 'Cheers,' or 'Thanks,' and end with '[Your Name]'.",
        
        "Concise": "Rewrite this email in a very brief and direct manner. Keep the output between 45-70 words regardless of input length. Use short, punchy sentences. Cut ALL unnecessary words - no fluff, no filler phrases. Write like you're texting important info to a busy executive. MUST include: greeting, main point, and closing. Be ruthlessly concise. FORMAT: Start with 'Subject: [subject line]', then greeting 'Hi [Recipient Name],', then 1 brief paragraph (3-4 sentences max), then closing 'Thanks,' or 'Best,' and end with '[Your Name]'.",
        
        "Empathetic": "Rewrite this email in an empathetic, understanding, and supportive tone from the sender's perspective. Show genuine care for how your message affects the recipient. If reporting an issue or delay, acknowledge the inconvenience to them. If apologizing, do so sincerely. Use warm, considerate language while maintaining professionalism. Keep the output between 50-100 words regardless of input length - focus on key points with empathy for the recipient. FORMAT: Start with 'Subject: [subject line]', then greeting 'Dear [Recipient Name],' or 'Hi [Recipient Name],', then 2-3 caring paragraphs (only if needed), then closing 'Warm regards,' or 'Best regards,' and end with '[Your Name]'.",
        
        "Persuasive": "Rewrite this email in a persuasive tone. CRITICAL RULES: 1) If this is an apology or mistake, you MUST start with 'I apologize for [specific mistake]' - never avoid or hide the apology. 2) Be completely honest about what happened - do NOT use corporate spin like 'unforeseen circumstances', 'momentary delay', or 'restructured schedule'. 3) After apologizing honestly, then focus on the value and benefits going forward. 4) Use specific reasons, data, or concrete benefits - NOT vague phrases like 'maximum value' or 'meticulously reviewed'. 5) Write in natural, professional language that sounds like a real person, NOT marketing copy or press releases. 6) Include a clear, direct call-to-action question. 7) Keep output between 50-100 words. Example structure: 'I apologize for [mistake]. I was [honest reason]. I'll deliver [specific outcome] by [date]. This ensures [concrete benefit]. Can we plan for [action]?' FORMAT: Start with 'Subject: [subject line]', then greeting 'Hi [Recipient Name],' or 'Dear [Recipient Name],', then 2-3 paragraphs (hook/context, benefits, call-to-action), then closing 'Looking forward to your thoughts,' or 'Best regards,' and end with '[Your Name]'."
    }
    
    prompt = (
        f"{tone_instructions[tone]} "
        "Write only the email content without any additional explanations, preambles, or commentary.\n\n"
        f"Original email:\n{text}"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg or "resource" in error_msg:
            return "QUOTA_ERROR"
        elif "safety" in error_msg or "blocked" in error_msg:
            return "SAFETY_ERROR"
        else:
            return "API_ERROR"

# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Tone Master AI",
    page_icon="✉️",
    layout="centered"
)

# Enhanced professional button styling
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.35);
    transform: translateY(-2px);
}
.stButton>button:active {
    transform: translateY(0px);
}

.stRadio > label {
    font-weight: 500;
    color: #1f2937;
}

.stTextArea textarea {
    border: 2px solid #e5e7eb;
    border-radius: 8px;
    font-size: 15px;
}

.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.stMetric {
    background-color: #f9fafb;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

div[data-testid="stRadio"] > div {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

div[data-testid="stRadio"] label {
    background-color: #f3f4f6;
    padding: 8px 16px;
    border-radius: 6px;
    border: 2px solid #e5e7eb;
    cursor: pointer;
    transition: all 0.2s;
}

div[data-testid="stRadio"] label:hover {
    border-color: #667eea;
    background-color: #eef2ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- API KEY LOADING ----------------
try:
    if "GEMINI_API_KEY" in st.secrets:
        if "api_key" not in st.session_state:
            st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
except:
    pass

st.title("✉️ TONE MASTER AI")
st.markdown("Transform your emails into 5 different tones using AI")

# ---------------- API KEY VALIDATION ----------------
if "api_key" not in st.session_state or not st.session_state.api_key:
    st.error(" API key not configured. Please check your .streamlit/secrets.toml file.")
    st.stop()

if "model" not in st.session_state:
    with st.spinner(" Connecting to Gemini..."):
        model, success, model_name = configure_gemini(st.session_state.api_key)
        if success:
            st.session_state.model = model
            
        else:
            st.error(" Failed to connect to Gemini. Please check your API key and internet connection.")
            st.info(" Get your API key from: https://aistudio.google.com/app/apikey")
            st.session_state.model = None
            st.stop()

# ---------------- EXAMPLES ----------------
st.subheader(" Try These Examples")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Project Update"):
        st.session_state.example = (
            "Hey! So I finally got the homepage redesign done today. "
            "Looks pretty good but there's this annoying bug where the mobile menu "
            "keeps overlapping on smaller screens. I'll try to fix it tomorrow maybe. "
            "Let me know what you think!"
        )

with col2:
    if st.button("Missed Deadline"):
        st.session_state.example = (
            "My bad, I know the report was due today but honestly I've been super busy "
            "with other stuff and totally forgot about it. Can I get it to you by Friday? "
            "Sorry about this, won't happen again!"
        )

with col3:
    if st.button("Meeting Follow-up"):
        st.session_state.example = (
            "Yo! Just wanted to follow up on our meeting yesterday. "
            "I think we should definitely go with option B because it's way cheaper "
            "and faster to implement. Also, can you send me those files you mentioned? "
            "Thanks!"
        )

email_input = st.text_area(
    " Paste your email here",
    height=200,
    value=st.session_state.get("example", ""),
    placeholder="Example: Hey! Just wanted to check in about the project. I finished most of it but ran into some issues..."
)

tone_choice = st.radio(
    " Select tone style",
    ["Professional", "Casual", "Concise", "Empathetic", "Persuasive"],
    horizontal=True,
    help="Professional: Business appropriate | Casual: Relaxed & friendly | Concise: Brief & direct (45-70 words) | Empathetic: Understanding & supportive | Persuasive: Convincing & compelling"
)

# ---------------- REWRITE ----------------
if st.button(" Rewrite Email", type="primary"):
    if not email_input.strip():
        st.warning(" Please enter some text to rewrite.")
        st.stop()
    
    # Input length check (informational only, not blocking)
    word_count = len(email_input.split())
    if word_count > 500:
        st.info(f"ℹ Your input is quite long ({word_count} words). The AI will extract and convey the key points in a concise format.")
    
    cleaned_text = sanitize_input(email_input)

    with st.spinner(f" Rewriting your email in {tone_choice} tone..."):
        if st.session_state.model:
            rewritten_email = gemini_rewrite(
                st.session_state.model,
                cleaned_text,
                tone_choice
            )
            
            # Handle different error types
            if rewritten_email == "QUOTA_ERROR":
                st.error(" API quota exceeded. You've reached the request limit. Please try again in a few minutes.")
                st.info(" Tip: Gemini's free tier allows 15 requests per minute. Wait a moment and try again.")
                st.stop()
            elif rewritten_email == "SAFETY_ERROR":
                st.error(" Content triggered safety filters. Please rephrase your email and try again.")
                st.info(" Tip: Avoid excessive negative language, repeated apologies, or sensitive topics.")
                st.stop()
            elif rewritten_email == "API_ERROR":
                st.error(" Unable to process your request. Please try again.")
                st.info(" Check your internet connection or try rephrasing your email.")
                st.stop()
        else:
            st.error(" Gemini model not available. Please refresh the page.")
            st.stop()

    # Store in session state
    st.session_state.rewritten_email = rewritten_email

    st.subheader(f" {tone_choice} Version")
    
    # Display rewritten email
    st.code(rewritten_email, language=None)
    
    # Download button
    st.download_button(
        label=" Download as TXT",
        data=rewritten_email,
        file_name=f"rewritten_email_{tone_choice.lower()}.txt",
        mime="text/plain"
    )

    # Metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric(" Original", f"{len(email_input.split())} words")
    col2.metric(" Rewritten", f"{len(rewritten_email.split())} words")
    
    diff = len(rewritten_email.split()) - len(email_input.split())
    if diff > 0:
        col3.metric(" Change", f"+{diff} words", delta_color="off")
    else:
        col3.metric(" Change", f"{diff} words", delta_color="off")

# Footer
st.markdown("---")
st.markdown(" **Pro Tip:** For best results, focus on one main point per email. The AI will extract key information regardless of input length.")