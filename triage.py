"""
Triage Module for Medical Chatbot

This module handles:
1. Triage classification (Emergency, Urgent, Routine)
2. Concern score calculation (0-10)
3. Specialist recommendation based on symptoms
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TriageLevel(Enum):
    """Triage levels for medical concerns."""
    EMERGENCY = "Emergency"
    URGENT = "Urgent"
    ROUTINE = "Routine"


# Emergency keywords - require immediate medical attention
EMERGENCY_KEYWORDS = [
    # Cardiac emergencies
    "chest pain", "heart attack", "cardiac arrest", "severe chest pressure",
    "crushing chest pain", "chest tightness", "chest pressure",
    
    # Respiratory emergencies
    "can't breathe", "cannot breathe", "severe shortness of breath",
    "difficulty breathing", "choking", "turning blue", "blue lips",
    "blue face", "cyanosis", "severe asthma attack",
    
    # Neurological emergencies
    "stroke", "unconscious", "passed out", "fainting", "seizure",
    "fits", "convulsions", "severe headache", "thunderclap headache",
    "worst headache", "sudden confusion", "can't speak", "slurred speech",
    "face drooping", "arm weakness", "vision loss", "sudden blindness",
    
    # Trauma emergencies
    "severe bleeding", "heavy bleeding", "bleeding won't stop",
    "spurting blood", "severed limb", "amputation", "penetrating wound",
    "gunshot", "stab wound", "head injury", "spinal injury",
    
    # Allergic emergencies
    "anaphylaxis", "severe allergic reaction", "swelling throat",
    "can't swallow", "difficulty swallowing", "tight throat",
    
    # Other emergencies
    "poisoning", "overdose", "suicide", "want to die", "kill myself",
    "severe burn", "electrical burn", "chemical burn", "near drowning",
    "heat stroke", "hypothermia", "shock", "cold clammy skin"
]

# Urgent keywords - need medical attention soon but not immediately life-threatening
URGENT_KEYWORDS = [
    # Fever
    "high fever", "persistent fever", "fever lasting", "fever for days",
    "temperature above 103", "temperature above 39.5",
    
    # Pain
    "severe pain", "intense pain", "unbearable pain", "worsening pain",
    "pain not improving", "persistent pain",
    
    # Gastrointestinal
    "blood in stool", "bloody diarrhea", "black tarry stool", "vomiting blood",
    "persistent vomiting", "can't keep food down", "severe dehydration",
    "severe abdominal pain", "rigid abdomen", "swollen abdomen",
    
    # Urinary
    "blood in urine", "can't urinate", "painful urination",
    
    # Respiratory
    "persistent cough", "coughing up blood", "wheezing", "persistent wheeze",
    "pneumonia symptoms", "bronchitis",
    
    # Other
    "sudden weight loss", "unexplained weight loss", "palpitations",
    "irregular heartbeat", "dizziness", "fainting spells",
    "severe weakness", "extreme fatigue", "confusion",
    "stiff neck", "rash with fever", "suspected fracture",
    "deep cut", "eye injury", "sudden vision changes"
]

# Routine keywords - can be managed with self-care or scheduled appointment
ROUTINE_KEYWORDS = [
    # Minor symptoms
    "mild headache", "tension headache", "minor headache",
    "common cold", "runny nose", "stuffy nose", "nasal congestion",
    "sneezing", "mild cough", "sore throat", "scratchy throat",
    "mild fever", "low grade fever",
    
    # Minor aches and pains
    "back pain", "muscle ache", "joint pain", "minor pain",
    "stiff neck", "muscle strain", "minor sprain",
    
    # Skin conditions
    "mild rash", "itchy skin", "dry skin", "acne", "pimple",
    "minor cut", "minor scrape", "minor burn", "sunburn",
    
    # Digestive
    "mild stomach ache", "indigestion", "heartburn", "constipation",
    "mild diarrhea", "gas", "bloating",
    
    # Other routine
    "allergy symptoms", "hay fever", "seasonal allergies",
    "insomnia", "trouble sleeping", "mild anxiety", "feeling down",
    "routine checkup", "follow up", "medication refill"
]

# Specialist mapping rules
SPECIALIST_RULES = [
    # Dermatology
    ("skin", "Dermatologist"),
    ("rash", "Dermatologist"),
    ("acne", "Dermatologist"),
    ("eczema", "Dermatologist"),
    ("psoriasis", "Dermatologist"),
    ("mole", "Dermatologist"),
    ("itchy skin", "Dermatologist"),
    ("hives", "Dermatologist"),
    ("dermatitis", "Dermatologist"),
    
    # Ophthalmology
    ("eye", "Ophthalmologist"),
    ("vision", "Ophthalmologist"),
    ("blurry vision", "Ophthalmologist"),
    ("eye pain", "Ophthalmologist"),
    ("red eye", "Ophthalmologist"),
    ("eye discharge", "Ophthalmologist"),
    ("cataract", "Ophthalmologist"),
    ("glaucoma", "Ophthalmologist"),
    
    # Cardiology
    ("chest pain", "Cardiologist"),
    ("heart", "Cardiologist"),
    ("palpitations", "Cardiologist"),
    ("irregular heartbeat", "Cardiologist"),
    ("high blood pressure", "Cardiologist"),
    ("hypertension", "Cardiologist"),
    ("heart disease", "Cardiologist"),
    ("angina", "Cardiologist"),
    
    # Pulmonology
    ("cough", "Pulmonologist"),
    ("asthma", "Pulmonologist"),
    ("copd", "Pulmonologist"),
    ("shortness of breath", "Pulmonologist"),
    ("wheezing", "Pulmonologist"),
    ("pneumonia", "Pulmonologist"),
    ("bronchitis", "Pulmonologist"),
    ("lung", "Pulmonologist"),
    
    # Gastroenterology
    ("stomach", "Gastroenterologist"),
    ("abdominal pain", "Gastroenterologist"),
    ("nausea", "Gastroenterologist"),
    ("vomiting", "Gastroenterologist"),
    ("diarrhea", "Gastroenterologist"),
    ("constipation", "Gastroenterologist"),
    ("heartburn", "Gastroenterologist"),
    ("acid reflux", "Gastroenterologist"),
    ("ibs", "Gastroenterologist"),
    ("ulcer", "Gastroenterologist"),
    ("gallbladder", "Gastroenterologist"),
    ("liver", "Gastroenterologist"),
    
    # Neurology
    ("headache", "Neurologist"),
    ("migraine", "Neurologist"),
    ("seizure", "Neurologist"),
    ("numbness", "Neurologist"),
    ("tingling", "Neurologist"),
    ("tremor", "Neurologist"),
    ("dizziness", "Neurologist"),
    ("vertigo", "Neurologist"),
    ("memory loss", "Neurologist"),
    ("multiple sclerosis", "Neurologist"),
    ("parkinson", "Neurologist"),
    
    # Orthopedics
    ("back pain", "Orthopedist"),
    ("joint pain", "Orthopedist"),
    ("fracture", "Orthopedist"),
    ("broken bone", "Orthopedist"),
    ("sprain", "Orthopedist"),
    ("strain", "Orthopedist"),
    ("arthritis", "Orthopedist"),
    ("knee pain", "Orthopedist"),
    ("shoulder pain", "Orthopedist"),
    ("hip pain", "Orthopedist"),
    ("spine", "Orthopedist"),
    
    # ENT (Ear, Nose, Throat)
    ("ear pain", "ENT Specialist"),
    ("ear infection", "ENT Specialist"),
    ("hearing loss", "ENT Specialist"),
    ("sinus", "ENT Specialist"),
    ("nasal", "ENT Specialist"),
    ("throat", "ENT Specialist"),
    ("tonsils", "ENT Specialist"),
    ("hoarseness", "ENT Specialist"),
    
    # Urology
    ("urinary", "Urologist"),
    ("kidney", "Urologist"),
    ("bladder", "Urologist"),
    ("prostate", "Urologist"),
    ("kidney stone", "Urologist"),
    ("blood in urine", "Urologist"),
    
    # Gynecology
    ("menstrual", "Gynecologist"),
    ("period", "Gynecologist"),
    ("pregnancy", "Gynecologist"),
    ("vaginal", "Gynecologist"),
    ("pelvic pain", "Gynecologist"),
    ("menopause", "Gynecologist"),
    ("pap smear", "Gynecologist"),
    
    # Psychiatry
    ("anxiety", "Psychiatrist"),
    ("depression", "Psychiatrist"),
    ("panic attack", "Psychiatrist"),
    ("bipolar", "Psychiatrist"),
    ("schizophrenia", "Psychiatrist"),
    ("suicidal", "Psychiatrist"),
    ("insomnia", "Psychiatrist"),
    ("eating disorder", "Psychiatrist"),
    ("ptsd", "Psychiatrist"),
    
    # Endocrinology
    ("diabetes", "Endocrinologist"),
    ("thyroid", "Endocrinologist"),
    ("hormone", "Endocrinologist"),
    ("hyperthyroidism", "Endocrinologist"),
    ("hypothyroidism", "Endocrinologist"),
    
    # Oncology
    ("cancer", "Oncologist"),
    ("tumor", "Oncologist"),
    ("mass", "Oncologist"),
    ("lump", "Oncologist"),
    ("chemotherapy", "Oncologist"),
    
    # Pediatrics (for children-specific queries)
    ("child", "Pediatrician"),
    ("baby", "Pediatrician"),
    ("infant", "Pediatrician"),
    ("toddler", "Pediatrician"),
    ("newborn", "Pediatrician"),
]


@dataclass
class TriageResult:
    """Result of triage classification."""
    triage_level: TriageLevel
    concern_score: int
    specialist: str
    matched_keywords: List[str]
    recommendation: str


def classify_triage(user_question: str) -> Tuple[TriageLevel, List[str]]:
    """
    Classify the triage level based on user query.
    
    Args:
        user_question: User's symptom description
        
    Returns:
        Tuple of (TriageLevel, matched_keywords)
    """
    q = user_question.lower()
    matched_keywords = []
    
    # Check for emergency keywords first (highest priority)
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in q:
            matched_keywords.append(keyword)
    
    if matched_keywords:
        return TriageLevel.EMERGENCY, matched_keywords
    
    # Check for urgent keywords
    for keyword in URGENT_KEYWORDS:
        if keyword in q:
            matched_keywords.append(keyword)
    
    if matched_keywords:
        return TriageLevel.URGENT, matched_keywords
    
    # Check for routine keywords
    for keyword in ROUTINE_KEYWORDS:
        if keyword in q:
            matched_keywords.append(keyword)
    
    # Default to routine if no keywords matched
    return TriageLevel.ROUTINE, matched_keywords


def calculate_concern_score(triage_level: TriageLevel, 
                            matched_keywords: List[str]) -> int:
    """
    Calculate concern score (0-10) based on triage level.
    
    Args:
        triage_level: The classified triage level
        matched_keywords: List of matched keywords
        
    Returns:
        Concern score (0-10)
    """
    base_scores = {
        TriageLevel.EMERGENCY: 9,
        TriageLevel.URGENT: 6,
        TriageLevel.ROUTINE: 2
    }
    
    base_score = base_scores.get(triage_level, 2)
    
    # Adjust score based on number of matched keywords
    keyword_count = len(matched_keywords)
    
    if triage_level == TriageLevel.EMERGENCY:
        # More emergency keywords = higher score
        if keyword_count >= 3:
            return 10
        elif keyword_count == 2:
            return 9
        else:
            return 8
    
    elif triage_level == TriageLevel.URGENT:
        # More urgent keywords = higher score
        if keyword_count >= 3:
            return 7
        elif keyword_count == 2:
            return 6
        else:
            return 5
    
    else:  # ROUTINE
        # Even routine can have some concern if multiple symptoms
        if keyword_count >= 3:
            return 4
        elif keyword_count == 2:
            return 3
        else:
            return 2


def suggest_specialist(user_question: str) -> str:
    """
    Suggest appropriate specialist based on symptoms.
    
    Args:
        user_question: User's symptom description
        
    Returns:
        Recommended specialist type
    """
    q = user_question.lower()
    
    # Check each specialist rule
    for keyword, specialist in SPECIALIST_RULES:
        if keyword in q:
            return specialist
    
    # Default to general physician
    return "General Physician"


def get_recommendation(triage_level: TriageLevel, 
                       concern_score: int,
                       specialist: str) -> str:
    """
    Generate recommendation based on triage results.
    
    Args:
        triage_level: The classified triage level
        concern_score: The calculated concern score
        specialist: Recommended specialist
        
    Returns:
        Recommendation text
    """
    if triage_level == TriageLevel.EMERGENCY:
        return (
            f"🚨 EMERGENCY: Call emergency services (911) immediately or go to the nearest emergency room. "
            f"Your symptoms indicate a potentially life-threatening condition. "
            f"Do not drive yourself. If possible, see a {specialist} after stabilization."
        )
    
    elif triage_level == TriageLevel.URGENT:
        return (
            f"⚠️ URGENT: You should seek medical care today. "
            f"Visit an urgent care center or contact your doctor for a same-day appointment. "
            f"Consider seeing a {specialist} for specialized evaluation."
        )
    
    else:  # ROUTINE
        return (
            f"✅ ROUTINE: Your symptoms appear to be non-urgent. "
            f"You can schedule a routine appointment with a {specialist}. "
            f"If symptoms worsen or new symptoms develop, seek medical care sooner."
        )


def analyze_symptoms(user_question: str) -> TriageResult:
    """
    Complete symptom analysis: triage, concern score, and specialist recommendation.
    
    Args:
        user_question: User's symptom description
        
    Returns:
        TriageResult with all analysis information
    """
    # Classify triage level
    triage_level, matched_keywords = classify_triage(user_question)
    
    # Calculate concern score
    concern_score = calculate_concern_score(triage_level, matched_keywords)
    
    # Suggest specialist
    specialist = suggest_specialist(user_question)
    
    # Generate recommendation
    recommendation = get_recommendation(triage_level, concern_score, specialist)
    
    return TriageResult(
        triage_level=triage_level,
        concern_score=concern_score,
        specialist=specialist,
        matched_keywords=matched_keywords,
        recommendation=recommendation
    )


# Convenience functions for direct use
def get_triage_label(user_question: str) -> str:
    """Get triage label as string."""
    triage_level, _ = classify_triage(user_question)
    return triage_level.value


def get_concern_score(user_question: str) -> int:
    """Get concern score (0-10)."""
    triage_level, matched_keywords = classify_triage(user_question)
    return calculate_concern_score(triage_level, matched_keywords)


def get_specialist(user_question: str) -> str:
    """Get recommended specialist."""
    return suggest_specialist(user_question)


# Test function
if __name__ == "__main__":
    print("Testing Triage Module...")
    print("=" * 70)
    
    test_cases = [
        "I have severe chest pain and can't breathe",
        "My baby has a high fever for 3 days",
        "I have a mild headache and runny nose",
        "I'm having a seizure and my vision is blurry",
        "I have a rash on my arm that's itchy",
        "I feel very anxious and can't sleep",
        "I have blood in my stool and severe abdominal pain",
    ]
    
    for test in test_cases:
        print(f"\nQuery: {test}")
        print("-" * 70)
        
        result = analyze_symptoms(test)
        
        print(f"Triage Level: {result.triage_level.value}")
        print(f"Concern Score: {result.concern_score}/10")
        print(f"Specialist: {result.specialist}")
        print(f"Matched Keywords: {result.matched_keywords}")
        print(f"Recommendation: {result.recommendation}")
        print("=" * 70)
