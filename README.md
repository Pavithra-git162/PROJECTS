# 📧 Tone Master AI – Email Tone Transformation System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

**Tone Master AI** is an AI-powered email rewriting system that transforms informal or unprofessional emails into polished corporate communication using **Google Gemini LLM and structured prompt engineering**.

---

## 🌐 Live Demo  
**Try it here:**  
🔗 *https://tone-master-ai.streamlit.app/*

---

##  Project Objective  
Writing professional emails is a challenge for students and professionals, and informal or unclear messages can negatively affect communication and professional image.

This project aims to build an AI-powered Email Tone Improver that automatically rewrites emails into different tone styles such as formal,professional,polite,and casual using Large Language Models(LLMs) to enhance clarity,tone appropriateness, and communication quality.

## 💡 Key Features  

###  AI Tone Transformation  
- Rewrites emails using **Google Gemini 2.5 Flash LLM**
- Supports **5 tone styles**:
  - Professional  
  - Casual  
  - Concise  
  - Empathetic  
  - Persuasive  

###  NLP Preprocessing  
- Regex-based sanitization of informal phrases  
- Converts slang into professional language before AI processing  

###  Structured Prompt Engineering  
- Enforces corporate email format:
  - Subject line  
  - Greeting  
  - Body paragraphs  
  - Closing signature  

###  Interactive Streamlit UI  
- Custom styled buttons and UI components  
- Example email templates  
- Tone selection radio buttons  

###  Error Handling  
- API quota error detection  
- Safety filter handling  
- Network/API failure handling  

###  Output Utilities  
- Word count analytics  
- Download rewritten email as `.txt` file  

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








