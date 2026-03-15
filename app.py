"""
Multimodal RAG Medical Chatbot with Concern Score & Hospital Referral

Streamlit main application file.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MediBot - Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emergency-badge {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .urgent-badge {
        background-color: #ff8800;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .routine-badge {
        background-color: #00aa44;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .concern-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .score-high {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .score-medium {
        background-color: #ffe6cc;
        color: #cc6600;
    }
    .score-low {
        background-color: #ccffcc;
        color: #006600;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #f0f0f0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #555;
    }
    .hospital-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .clarification-box {
        background-color: #e7f3ff;
        border: 2px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if "hospitals_shown" not in st.session_state:
    st.session_state.hospitals_shown = False

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = False

if "last_location" not in st.session_state:
    st.session_state.last_location = ""


def check_api_keys():
    """Check if required API keys are configured."""
    google_key = os.getenv("GOOGLE_API_KEY")
    locationiq_key = os.getenv("LOCATIONIQ_API_KEY")
    
    google_ok = google_key and google_key != "your_google_api_key_here"
    
    return {
        "google": google_ok,
        "locationiq": bool(locationiq_key)
    }


def get_triage_badge_class(triage_level: str) -> str:
    """Get CSS class for triage badge."""
    if triage_level == "Emergency":
        return "emergency-badge"
    elif triage_level == "Urgent":
        return "urgent-badge"
    else:
        return "routine-badge"


def get_score_class(score: int) -> str:
    """Get CSS class for concern score."""
    if score >= 8:
        return "score-high"
    elif score >= 5:
        return "score-medium"
    else:
        return "score-low"


def get_confidence_badge_class(confidence_level: str) -> str:
    """Get CSS class for confidence badge."""
    if confidence_level == "high":
        return "confidence-high"
    elif confidence_level == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"


def display_chat_history():
    """Display chat history."""
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>{msg['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Display assistant message
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🏥 MediBot:</strong><br>{msg['content']}
                </div>
            """, unsafe_allow_html=True)
            
            # Display confidence if available
            if 'confidence' in msg and msg['confidence'] is not None:
                conf_class = get_confidence_badge_class(msg['confidence_level'])
                st.markdown(f"""
                    <div style="text-align: right; margin: 0.5rem 0;">
                        <span class="{conf_class}">
                            Confidence: {msg['confidence']:.0%} ({msg['confidence_level']})
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Only show triage info if not a clarification request
            if not msg.get('is_clarification', False):
                badge_class = get_triage_badge_class(msg['triage'])
                score_class = get_score_class(msg['score'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <span class="{badge_class}">{msg['triage']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="concern-score {score_class}">
                            Concern Score: {msg['score']}/10
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display specialist recommendation
                st.info(f"**Recommended Specialist:** {msg['specialist']}")
                
                # Display hospitals only if concern score > 5 and not already shown
                if msg.get('hospitals') and msg['score'] > 5:
                    st.subheader("🏥 Nearby Hospitals")
                    for h in msg['hospitals']:
                        distance_str = ""
                        if h.get('distance'):
                            distance_km = h['distance'] / 1000
                            distance_str = f" ({distance_km:.1f} km)"
                        
                        st.markdown(f"""
                            <div class="hospital-card">
                                <strong>{h['name']}</strong>{distance_str}<br>
                                📍 {h['address'][:100]}...
                            </div>
                        """, unsafe_allow_html=True)
                
                # Show recommendation based on triage
                if msg['triage'] == "Emergency":
                    st.error("""
                        🚨 **EMERGENCY DETECTED** 🚨\n\n
                        Your symptoms indicate a potentially life-threatening condition. 
                        **Call emergency services (911) immediately** or go to the nearest emergency room. 
                        Do not drive yourself.
                    """)
                elif msg['triage'] == "Urgent":
                    st.warning("""
                        ⚠️ **URGENT CARE NEEDED**\n\n
                        You should seek medical care today. Visit an urgent care center or contact your doctor for a same-day appointment.
                    """)
            
            st.markdown("---")


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🏥 MediBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Medical Assistant with Smart Triage & Hospital Referral</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This chatbot provides general health information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider. If you think you may have a medical emergency, call emergency services immediately.
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key inputs
        st.subheader("API Configuration")
        
        google_api_key = st.text_input(
            "Google API Key",
            type="password",
            value="",
            help="Get your key from https://makersuite.google.com/app/apikey"
        )
        
        locationiq_api_key = st.text_input(
            "LocationIQ API Key (for hospital search)",
            type="password",
            value="",
            help="Get free key from https://locationiq.com/"
        )
        
        # Set API keys
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if locationiq_api_key:
            os.environ["LOCATIONIQ_API_KEY"] = locationiq_api_key
        
        # Check API keys
        api_status = check_api_keys()
        
        if api_status["google"]:
            st.success("✅ Google API configured")
        else:
            st.error("⚠️ Google API key required")
        
        if api_status["locationiq"]:
            st.success("✅ LocationIQ API configured")
        else:
            st.info("ℹ️ LocationIQ API optional (for hospital search)")
        
        st.session_state.api_key_set = api_status["google"]
        
        st.markdown("---")
        
        # Location settings
        st.subheader("📍 Your Location")
        location = st.text_input(
            "City / Address",
            value="Chennai, India",
            help="Enter your city for hospital recommendations"
        )
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
            **MediBot** features:
            - 🔍 RAG-based medical information
            - 📊 Smart triage with concern scoring
            - 📈 Confidence rating for responses
            - 🏥 Location-based hospital finder
            - 💬 Follow-up questions for clarity
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.history = []
            st.session_state.hospitals_shown = False
            st.session_state.pending_clarification = False
            st.rerun()
    
    # Main content area
    if not st.session_state.api_key_set:
        st.warning("👈 Please configure your Google API key in the sidebar to use the chatbot.")
        st.info("Get your free API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Display chat history
    if st.session_state.history:
        display_chat_history()
    
    # Show clarification notice if pending
    if st.session_state.pending_clarification:
        st.info("📝 I'm waiting for more details about your symptoms. Please provide additional information so I can help you better.")
    
    # Chat input
    user_question = st.chat_input("Describe your symptoms or ask a health question...")
    
    if user_question:
        # Show user message immediately
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>{user_question}
            </div>
        """, unsafe_allow_html=True)
        
        # Process the query
        with st.spinner("Analyzing your query..."):
            try:
                # Import modules here to ensure API key is set
                from rag_pipeline import get_rag_answer
                from triage import analyze_symptoms
                from hospitals import get_nearby_hospitals
                
                # Get RAG answer with chat history for context
                rag_response = get_rag_answer(
                    user_question, 
                    chat_history=st.session_state.history
                )
                
                # Check if this is a clarification response
                is_clarification = rag_response.needs_more_info
                
                if is_clarification:
                    # Store clarification request
                    st.session_state.history.append({
                        "role": "user",
                        "content": user_question
                    })
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": rag_response.answer,
                        "confidence": rag_response.confidence,
                        "confidence_level": rag_response.confidence_level,
                        "is_clarification": True
                    })
                    st.session_state.pending_clarification = True
                    
                    # Display clarification request
                    st.markdown(f"""
                        <div class="clarification-box">
                            <strong>🏥 MediBot:</strong><br>{rag_response.answer}
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # Normal response flow
                    st.session_state.pending_clarification = False
                    
                    # Triage analysis
                    triage_result = analyze_symptoms(user_question)
                    triage_label = triage_result.triage_level.value
                    concern_score = triage_result.concern_score
                    specialist = triage_result.specialist
                    
                    # Get nearby hospitals only if concern score > 5 and not already shown
                    hospitals = []
                    if concern_score > 5 and not st.session_state.hospitals_shown:
                        if location:
                            hospitals = get_nearby_hospitals(location, specialist, max_results=5)
                            if hospitals:
                                st.session_state.hospitals_shown = True
                                st.session_state.last_location = location
                    
                    # Store in history
                    st.session_state.history.append({
                        "role": "user",
                        "content": user_question
                    })
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": rag_response.answer,
                        "confidence": rag_response.confidence,
                        "confidence_level": rag_response.confidence_level,
                        "triage": triage_label,
                        "score": concern_score,
                        "specialist": specialist,
                        "hospitals": hospitals,
                        "is_clarification": False
                    })
                    
                    # Display assistant response
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🏥 MediBot:</strong><br>{rag_response.answer}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence
                    conf_class = get_confidence_badge_class(rag_response.confidence_level)
                    st.markdown(f"""
                        <div style="text-align: right; margin: 0.5rem 0;">
                            <span class="{conf_class}">
                                Confidence: {rag_response.confidence:.0%} ({rag_response.confidence_level})
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display triage info
                    badge_class = get_triage_badge_class(triage_label)
                    score_class = get_score_class(concern_score)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <span class="{badge_class}">{triage_label}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                            <div class="concern-score {score_class}">
                                Concern Score: {concern_score}/10
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display specialist recommendation
                    st.info(f"**Recommended Specialist:** {specialist}")
                    
                    # Display hospitals only if concern score > 5
                    if concern_score > 5:
                        if hospitals:
                            st.subheader("🏥 Nearby Hospitals")
                            for h in hospitals:
                                distance_str = ""
                                if h.get('distance'):
                                    distance_km = h['distance'] / 1000
                                    distance_str = f" ({distance_km:.1f} km)"
                                
                                st.markdown(f"""
                                    <div class="hospital-card">
                                        <strong>{h['name']}</strong>{distance_str}<br>
                                        📍 {h['address'][:100]}...
                                    </div>
                                """, unsafe_allow_html=True)
                        elif location and check_api_keys()["locationiq"]:
                            st.warning("""
                                No hospitals found in your area. Please visit the nearest government hospital or consult an online telemedicine service.
                            """)
                    
                    # Show recommendation based on triage
                    if triage_label == "Emergency":
                        st.error("""
                            🚨 **EMERGENCY DETECTED** 🚨\n\n
                            Your symptoms indicate a potentially life-threatening condition. 
                            **Call emergency services (911) immediately** or go to the nearest emergency room. 
                            Do not drive yourself.
                        """)
                    elif triage_label == "Urgent":
                        st.warning("""
                            ⚠️ **URGENT CARE NEEDED**\n\n
                            You should seek medical care today. Visit an urgent care center or contact your doctor for a same-day appointment.
                        """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please make sure you have installed all required packages and set up your API keys correctly.")


if __name__ == "__main__":
    main()
