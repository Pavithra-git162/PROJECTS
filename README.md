#  Tone Master AI 
Tone Master AI is an AI-powered platform that rewrites emails into clear, concise, and context-aware communication, tailored to different tones and languages.


##  Live Demo   
🔗 *https://tone-master-ai.streamlit.app/*


##  Project Objective  
Writing professional emails is a challenge for students and professionals, and informal or unclear messages can negatively affect communication and professional image. This project aims to build an AI-powered Email Tone Improver that automatically rewrites emails into different tone and languages.


##  Features  
-  AI-powered email rewriting using Google Gemini (Gemini 2.5 Flash)
-  Multiple tone options (Professional, Casual, Concise, Apologetic, Follow-up)
-  Multi-language support (English, Tamil, Hindi)
-  Text similarity analysis using TF-IDF and cosine similarity
-  User-based email history storage using SQLite
-  Authentication system (Login / Signup)
-  Deployed on Streamlit Cloud for real-time access



 




---

## 🛠 Tech Stack  

| Category | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| AI Model | Google Gemini 2.5 Flash |
| NLP | Regex, Prompt Engineering |
| UI Styling | Streamlit built-in components |
| Version Control | Git & GitHub |
| Deployment | Streamlit Cloud |


---
## 📂 Project Structure  

```bash
Tone-Master-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
```

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

# HOW THE SYSTEM WORKS
## Enter Email Content & Select the tone

<img width="1080" height="897" alt="ui" src="https://github.com/user-attachments/assets/040df517-242f-4f1d-9202-dbd4c6e8388a" />

## AI-Generated Email Output

<img width="929" height="659" alt="Screenshot 2026-02-06 224209" src="https://github.com/user-attachments/assets/28019260-6ba7-4289-b369-109087831362" />

# 🧩 FUTURE IMPROVEMENTS

## Multi-Language Support  
Add support for multiple languages (Tamil, English, Hindi, etc.) using multilingual LLM models to rewrite emails in different languages.

## Plagiarism / Similarity Score 
Implement semantic similarity and plagiarism detection using cosine similarity or embedding models to measure how much the rewritten email differs from the original text.
  
## User Authentication  
Add login/signup functionality so users can maintain personal email history and preferences.

## Email History & Save Feature  
Store user-generated emails in a database or local storage so users can view, edit, and reuse previous emails.










