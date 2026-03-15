"""
Evaluation Script for Medical Chatbot

This script evaluates:
1. Triage classification accuracy
2. Specialist recommendation accuracy
3. Hospital suggestion relevance
"""

import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from triage import analyze_symptoms, classify_triage, suggest_specialist, TriageLevel
from hospitals import get_nearby_hospitals


@dataclass
class TestCase:
    """Test case for evaluation."""
    query: str
    expected_triage: str
    expected_specialist: str
    description: str


# Test cases for evaluation
TEST_CASES = [
    # Emergency cases
    TestCase(
        query="I have severe chest pain and can't breathe",
        expected_triage="Emergency",
        expected_specialist="Cardiologist",
        description="Cardiac emergency"
    ),
    TestCase(
        query="My father is unconscious and not responding",
        expected_triage="Emergency",
        expected_specialist="Neurologist",
        description="Unconsciousness"
    ),
    TestCase(
        query="I'm having a seizure and my vision is blurry",
        expected_triage="Emergency",
        expected_specialist="Neurologist",
        description="Seizure"
    ),
    TestCase(
        query="Severe bleeding from my arm that won't stop",
        expected_triage="Emergency",
        expected_specialist="General Physician",
        description="Severe bleeding"
    ),
    TestCase(
        query="My throat is swelling and I can't breathe properly",
        expected_triage="Emergency",
        expected_specialist="ENT Specialist",
        description="Anaphylaxis"
    ),
    
    # Urgent cases
    TestCase(
        query="My baby has a high fever for 3 days",
        expected_triage="Urgent",
        expected_specialist="Pediatrician",
        description="Pediatric fever"
    ),
    TestCase(
        query="I have blood in my stool and severe abdominal pain",
        expected_triage="Urgent",
        expected_specialist="Gastroenterologist",
        description="GI bleeding"
    ),
    TestCase(
        query="Persistent cough with blood for a week",
        expected_triage="Urgent",
        expected_specialist="Pulmonologist",
        description="Hemoptysis"
    ),
    TestCase(
        query="High fever of 104 degrees for 2 days",
        expected_triage="Urgent",
        expected_specialist="General Physician",
        description="High fever"
    ),
    TestCase(
        query="Severe headache with stiff neck and fever",
        expected_triage="Urgent",
        expected_specialist="Neurologist",
        description="Meningitis symptoms"
    ),
    
    # Routine cases
    TestCase(
        query="I have a mild headache and runny nose",
        expected_triage="Routine",
        expected_specialist="General Physician",
        description="Common cold"
    ),
    TestCase(
        query="Acne on my face that's not going away",
        expected_triage="Routine",
        expected_specialist="Dermatologist",
        description="Acne"
    ),
    TestCase(
        query="Mild back pain after lifting heavy boxes",
        expected_triage="Routine",
        expected_specialist="Orthopedist",
        description="Back pain"
    ),
    TestCase(
        query="Dry skin and mild itching on my arms",
        expected_triage="Routine",
        expected_specialist="Dermatologist",
        description="Dry skin"
    ),
    TestCase(
        query="I feel a bit anxious and have trouble sleeping",
        expected_triage="Routine",
        expected_specialist="Psychiatrist",
        description="Anxiety"
    ),
    TestCase(
        query="My vision is a bit blurry when reading",
        expected_triage="Routine",
        expected_specialist="Ophthalmologist",
        description="Vision problems"
    ),
    TestCase(
        query="Mild stomach ache after eating spicy food",
        expected_triage="Routine",
        expected_specialist="Gastroenterologist",
        description="Indigestion"
    ),
    TestCase(
        query="Joint pain in my knees when walking",
        expected_triage="Routine",
        expected_specialist="Orthopedist",
        description="Joint pain"
    ),
]


def evaluate_triage(test_cases: List[TestCase]) -> Dict:
    """
    Evaluate triage classification accuracy.
    
    Args:
        test_cases: List of test cases
        
    Returns:
        Evaluation metrics dictionary
    """
    print("=" * 70)
    print("TRIAGE CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    correct = 0
    confusion_matrix = {
        "Emergency": {"Emergency": 0, "Urgent": 0, "Routine": 0},
        "Urgent": {"Emergency": 0, "Urgent": 0, "Routine": 0},
        "Routine": {"Emergency": 0, "Urgent": 0, "Routine": 0}
    }
    
    results = []
    
    for test in test_cases:
        # Get predicted triage
        triage_level, matched_keywords = classify_triage(test.query)
        predicted = triage_level.value
        
        # Check if correct
        is_correct = predicted == test.expected_triage
        if is_correct:
            correct += 1
        
        # Update confusion matrix
        confusion_matrix[test.expected_triage][predicted] += 1
        
        # Store result
        results.append({
            "query": test.query,
            "expected": test.expected_triage,
            "predicted": predicted,
            "correct": is_correct,
            "keywords": matched_keywords
        })
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {test.description}")
        print(f"   Query: {test.query}")
        print(f"   Expected: {test.expected_triage} | Predicted: {predicted}")
        if matched_keywords:
            print(f"   Matched keywords: {matched_keywords}")
    
    # Calculate metrics
    accuracy = correct / len(test_cases) if test_cases else 0
    
    print("\n" + "=" * 70)
    print("TRIAGE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"{'Expected/Predicted':<20} {'Emergency':<10} {'Urgent':<10} {'Routine':<10}")
    print("-" * 40)
    for expected in ["Emergency", "Urgent", "Routine"]:
        row = confusion_matrix[expected]
        print(f"{expected:<20} {row['Emergency']:<10} {row['Urgent']:<10} {row['Routine']:<10}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "confusion_matrix": confusion_matrix,
        "results": results
    }


def evaluate_specialist(test_cases: List[TestCase]) -> Dict:
    """
    Evaluate specialist recommendation accuracy.
    
    Args:
        test_cases: List of test cases
        
    Returns:
        Evaluation metrics dictionary
    """
    print("\n" + "=" * 70)
    print("SPECIALIST RECOMMENDATION EVALUATION")
    print("=" * 70)
    
    correct = 0
    results = []
    
    for test in test_cases:
        # Get predicted specialist
        predicted = suggest_specialist(test.query)
        
        # Check if correct (allow partial match)
        is_correct = (
            predicted.lower() == test.expected_specialist.lower() or
            test.expected_specialist.lower() in predicted.lower() or
            predicted.lower() in test.expected_specialist.lower()
        )
        
        if is_correct:
            correct += 1
        
        # Store result
        results.append({
            "query": test.query,
            "expected": test.expected_specialist,
            "predicted": predicted,
            "correct": is_correct
        })
        
        # Print result
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {test.description}")
        print(f"   Query: {test.query}")
        print(f"   Expected: {test.expected_specialist} | Predicted: {predicted}")
    
    # Calculate metrics
    accuracy = correct / len(test_cases) if test_cases else 0
    
    print("\n" + "=" * 70)
    print("SPECIALIST EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "results": results
    }


def evaluate_hospitals() -> Dict:
    """
    Evaluate hospital recommendation functionality.
    
    Returns:
        Evaluation metrics dictionary
    """
    print("\n" + "=" * 70)
    print("HOSPITAL RECOMMENDATION EVALUATION")
    print("=" * 70)
    
    test_cases = [
        ("Chennai", "Cardiologist"),
        ("Chennai", "Dermatologist"),
        ("Coimbatore", "General Physician"),
        ("Madurai", "Neurologist"),
        ("Trichy", "Orthopedist"),
        ("Unknown City", "Cardiologist"),  # Test fallback
    ]
    
    results = []
    
    for city, specialist in test_cases:
        hospitals = get_nearby_hospitals(city, specialist, max_results=3)
        
        has_results = len(hospitals) > 0
        correct_city = all(h['city'].lower() == city.lower() for h in hospitals) if hospitals else False
        
        results.append({
            "city": city,
            "specialist": specialist,
            "hospitals_found": len(hospitals),
            "has_results": has_results,
            "correct_city": correct_city
        })
        
        status = "✅" if has_results else "⚠️"
        print(f"\n{status} City: {city}, Specialist: {specialist}")
        print(f"   Hospitals found: {len(hospitals)}")
        for h in hospitals:
            print(f"   - {h['name']} ({h['speciality']})")
    
    total_tests = len(test_cases)
    successful_tests = sum(1 for r in results if r['has_results'])
    
    print("\n" + "=" * 70)
    print("HOSPITAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {total_tests}")
    print(f"Successful queries: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests:.2%}")
    
    return {
        "success_rate": successful_tests / total_tests,
        "successful": successful_tests,
        "total": total_tests,
        "results": results
    }


def generate_report(triage_eval: Dict, specialist_eval: Dict, hospital_eval: Dict) -> str:
    """
    Generate evaluation report.
    
    Args:
        triage_eval: Triage evaluation results
        specialist_eval: Specialist evaluation results
        hospital_eval: Hospital evaluation results
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("MEDICAL CHATBOT EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("1. TRIAGE CLASSIFICATION")
    report.append("-" * 40)
    report.append(f"Accuracy: {triage_eval['accuracy']:.2%}")
    report.append(f"Correct: {triage_eval['correct']}/{triage_eval['total']}")
    report.append("")
    report.append("Confusion Matrix:")
    cm = triage_eval['confusion_matrix']
    for expected in ["Emergency", "Urgent", "Routine"]:
        row = cm[expected]
        report.append(f"  {expected}: Emergency={row['Emergency']}, Urgent={row['Urgent']}, Routine={row['Routine']}")
    
    report.append("")
    report.append("2. SPECIALIST RECOMMENDATION")
    report.append("-" * 40)
    report.append(f"Accuracy: {specialist_eval['accuracy']:.2%}")
    report.append(f"Correct: {specialist_eval['correct']}/{specialist_eval['total']}")
    
    report.append("")
    report.append("3. HOSPITAL RECOMMENDATION")
    report.append("-" * 40)
    report.append(f"Success Rate: {hospital_eval['success_rate']:.2%}")
    report.append(f"Successful: {hospital_eval['successful']}/{hospital_eval['total']}")
    
    report.append("")
    report.append("=" * 70)
    report.append("OVERALL SUMMARY")
    report.append("=" * 70)
    overall = (triage_eval['accuracy'] + specialist_eval['accuracy'] + hospital_eval['success_rate']) / 3
    report.append(f"Overall Performance: {overall:.2%}")
    report.append("")
    
    return "\n".join(report)


def save_results(triage_eval: Dict, specialist_eval: Dict, hospital_eval: Dict, filename: str = "evaluation_results.json"):
    """
    Save evaluation results to JSON file.
    
    Args:
        triage_eval: Triage evaluation results
        specialist_eval: Specialist evaluation results
        hospital_eval: Hospital evaluation results
        filename: Output filename
    """
    results = {
        "triage_evaluation": triage_eval,
        "specialist_evaluation": specialist_eval,
        "hospital_evaluation": hospital_eval,
        "overall_accuracy": (triage_eval['accuracy'] + specialist_eval['accuracy'] + hospital_eval['success_rate']) / 3
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def main():
    """Main evaluation function."""
    print("\n" + "=" * 70)
    print("MEDICAL CHATBOT EVALUATION")
    print("=" * 70)
    print("\nThis script evaluates the chatbot's triage, specialist, and hospital")
    print("recommendation systems using predefined test cases.")
    print("")
    
    # Run evaluations
    triage_eval = evaluate_triage(TEST_CASES)
    specialist_eval = evaluate_specialist(TEST_CASES)
    hospital_eval = evaluate_hospitals()
    
    # Generate and print report
    report = generate_report(triage_eval, specialist_eval, hospital_eval)
    print("\n" + report)
    
    # Save results
    save_results(triage_eval, specialist_eval, hospital_eval)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
