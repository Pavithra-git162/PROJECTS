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


##  Tech Stack  
### Frontend UI
 - Streamlit UI
   
### Backend
 - Python
  
### Database
- SQLite

### Machine Learning/NLP
- Scikit-learn

### AI Integration
- Google Gemini API(Gemini 2.5 Flash)
   

##  Project Structure  
Tone-Master-AI/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


## Installation(Local Setup)
### Clone the repository
- git clone https://github.com/your-username/Tone-Master-AI.git
- cd Tone-Master-AI

### Install dependencies
- pip install -r requirements.txt

### Run the app
- streamlit run app.py

## Environment Setup
 Create .streamlit/secrets.toml and add:
- GEMINI_API_KEY = "your_api_key_here"
  
## Future Enhancements 
- Integration with email platforms (Gmail/Outlook)
- Voice-to-text input 
















