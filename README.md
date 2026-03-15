# 🏥 MediBot - Multimodal RAG Medical Chatbot

An AI-powered medical assistant with smart triage, confidence scoring, and location-based hospital referrals.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys](#api-keys)
- [Disclaimer](#disclaimer)

## 🔍 Overview

MediBot is a Retrieval-Augmented Generation (RAG) based medical chatbot that provides:
- **Accurate medical information** using vector similarity search
- **Smart triage classification** (Emergency/Urgent/Routine)
- **Concern scoring** (0-10 scale)
- **Confidence rating** for response accuracy
- **Follow-up questions** when confidence is low
- **Location-based hospital suggestions** using LocationIQ API

## ✨ Features

### 1. RAG-Based Medical Information
- Uses FAISS vector store for efficient document retrieval
- Google Gemini LLM for response generation
- Medical corpus covering cardiac, respiratory, neurological, and other conditions

### 2. Smart Triage System
- **Emergency (8-10)**: Life-threatening conditions requiring immediate care
- **Urgent (5-7)**: Conditions needing same-day medical attention
- **Routine (1-4)**: Non-urgent conditions suitable for scheduled appointments

### 3. Confidence Scoring
- **High (≥70%)**: Response is reliable based on retrieved documents
- **Medium (40-69%)**: Response is reasonably accurate
- **Low (<40%)**: System asks for more symptoms to improve accuracy

### 4. Chat History Context
- When confidence is low, the system asks for more details
- Follow-up responses use previous chat history for better context

### 5. Hospital Finder (LocationIQ API)
- **Only shows hospitals when concern score > 5**
- Shows hospitals **once per chat session**
- Location-based search using real hospital data
- Distance information for each hospital

## 🏗️ Architecture

```
User Query → Embeddings + FAISS Retrieval → Confidence Calculation
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Confidence HIGH/              Confidence LOW
            MEDIUM                          ↓
                    ↓                       Ask for more
            Generate Response                 symptoms
                    ↓                               ↓
            Triage Classification ←── Use chat history
                    ↓
            Concern Score > 5?
                    ↓
            Yes → Show hospitals (once)
            No  → Skip hospitals
```

### Components

| Component | Description |
|-----------|-------------|
| `app.py` | Streamlit UI with confidence display and hospital logic |
| `rag_pipeline.py` | RAG with confidence scoring and chat history |
| `triage.py` | Triage classification and concern scoring |
| `hospitals.py` | LocationIQ API integration for hospital search |
| `build_index.py` | FAISS index creation script |

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Google Gemini API key
- LocationIQ API key (free tier available)

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd medical_rag_chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key_here
LOCATIONIQ_API_KEY=your_locationiq_api_key_here
```

Get your API keys:
- **Google API**: https://makersuite.google.com/app/apikey
- **LocationIQ**: https://locationiq.com/ (free tier: 10,000 requests/day)

### Step 4: Build the Vector Index

```bash
python build_index.py
```

## 💻 Usage

### Run the Chatbot

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Enter your location** in the sidebar (City/Address)
2. **Type your symptoms** or health question
3. **Review the response** which includes:
   - AI-generated medical information
   - **Confidence rating** (High/Medium/Low)
   - Triage classification (Emergency/Urgent/Routine)
   - Concern score (0-10)
   - Recommended specialist
   - Nearby hospitals (only if concern score > 5)

### Low Confidence Handling

If the system has **low confidence** in understanding your query:
- It will ask for more details (symptoms, duration, severity, etc.)
- Your follow-up response will be analyzed with chat history context
- This improves the accuracy of the response

### Example Queries

**High Confidence (Detailed):**
- "I have severe chest pain and difficulty breathing for the past 2 hours"
- "My child has a high fever of 103°F for 3 days with cough"

**Low Confidence (Vague):**
- "I feel sick" → System asks for more details
- "Help" → System asks for symptoms

## 📁 Project Structure

```
medical_rag_chatbot/
├── app.py                  # Streamlit main application
├── rag_pipeline.py         # RAG pipeline with confidence scoring
├── triage.py               # Triage and concern scoring
├── hospitals.py            # LocationIQ hospital finder
├── build_index.py          # FAISS index builder
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not in git)
├── .env.example            # API key template
├── README.md               # This file
├── data/
│   └── webmd_texts/        # Medical corpus (8 files)
└── vector_store/
    └── faiss_index/        # Saved FAISS index
```

## 🔑 API Keys

### Google Gemini API Key

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

### LocationIQ API Key (Free)

1. Visit https://locationiq.com/
2. Sign up for a free account
3. Get your API key from the dashboard
4. Copy the key to your `.env` file

**Free tier includes:**
- 10,000 requests/day
- Perfect for testing and small deployments

## 🎯 Key Behaviors

### Hospital Display Logic

Hospitals are **only shown when:**
1. Concern score > 5 (Urgent or Emergency)
2. Hospitals haven't been shown yet in current chat
3. LocationIQ API is configured

### Confidence Calculation

Confidence is based on:
- **Document similarity** (50%): How well retrieved documents match the query
- **Query specificity** (30%): Length and detail of the query
- **Medical terms** (20%): Presence of relevant medical keywords

### Chat History Usage

Chat history is used when:
- Confidence is low and user provides follow-up information
- Previous context helps improve response accuracy

## 📊 Confidence Levels

| Level | Range | Behavior |
|-------|-------|----------|
| **High** | ≥70% | Direct medical response |
| **Medium** | 40-69% | Direct response with standard disclaimer |
| **Low** | <40% | Ask for more symptoms/details |

## 🛠️ Customization

### Adjusting Confidence Thresholds

Edit `rag_pipeline.py`:
```python
HIGH_CONFIDENCE = 0.7    # Change to your preferred threshold
MEDIUM_CONFIDENCE = 0.4  # Change to your preferred threshold
```

### Modifying Triage Rules

Edit `triage.py` to customize:
- `EMERGENCY_KEYWORDS`: Emergency symptom keywords
- `URGENT_KEYWORDS`: Urgent symptom keywords
- `SPECIALIST_RULES`: Specialist mapping rules

### Adding Medical Content

1. Add text files to `data/webmd_texts/`
2. Rebuild the index: `python build_index.py`

## 📚 Dependencies

- `streamlit`: Web application framework
- `langchain`: LLM orchestration
- `langchain-google-genai`: Google Gemini integration
- `faiss-cpu`: Vector similarity search
- `google-generativeai`: Google AI API
- `requests`: HTTP requests for LocationIQ API
- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations for confidence calculation

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This chatbot provides **general health information only** and is **NOT a substitute for professional medical advice, diagnosis, or treatment**.

- Always seek the advice of your physician or other qualified health provider
- Never disregard professional medical advice or delay seeking it because of information from this chatbot
- If you think you may have a medical emergency, **call emergency services (911) immediately**
- Do not use this chatbot for emergency situations requiring immediate medical attention

## 🙏 Acknowledgments

- Google Gemini for LLM capabilities
- LangChain for RAG framework
- FAISS for vector search
- LocationIQ for geocoding and hospital data
- Streamlit for the web interface
