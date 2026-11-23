"""
End-to-End Verification Test Script for Healthcare AI Prototype.
Tests:
1. Inference pipeline prediction on various clinical profiles.
2. In-domain RAG retrieval and citation.
3. Out-of-domain RAG guardrail refusal.
"""

from src.train_ml import predict_patient_risk
from src.rag_engine import RAGEngine


def run_tests():
    print("=" * 70)
    print("RUNNING END-TO-END VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test ML Inference Pipeline
    print("\n--- 1. Testing ML Inference Pipeline ---")
    normal_patient = {
        "Pregnancies": 1,
        "Glucose": 85.0,
        "BloodPressure": 68.0,
        "SkinThickness": 20.0,
        "Insulin": 70.0,
        "BMI": 22.0,
        "DiabetesPedigreeFunction": 0.20,
        "Age": 24,
    }
    high_risk_patient = {
        "Pregnancies": 5,
        "Glucose": 180.0,
        "BloodPressure": 90.0,
        "SkinThickness": 35.0,
        "Insulin": 220.0,
        "BMI": 38.0,
        "DiabetesPedigreeFunction": 0.85,
        "Age": 55,
    }

    res_normal = predict_patient_risk(normal_patient)
    res_high = predict_patient_risk(high_risk_patient)

    print(f"Normal Patient: Risk = {res_normal['risk_percentage']}% | Tier = {res_normal['risk_tier']} | Label = {res_normal['predicted_label']}")
    print(f"High Risk Patient: Risk = {res_high['risk_percentage']}% | Tier = {res_high['risk_tier']} | Label = {res_high['predicted_label']}")
    print(f"High Risk Clinical Flags: {res_high['clinical_flags']}")

    assert res_normal["risk_probability"] < res_high["risk_probability"], "Risk ranking failed!"
    print("✅ ML Inference Pipeline tests passed.")

    # 2. Test Grounded RAG Engine
    print("\n--- 2. Testing Grounded RAG Engine ---")
    rag = RAGEngine()

    # In-domain query 1
    q_in1 = "What are the classic symptoms of diabetes?"
    r_in1 = rag.ask(q_in1)
    print(f"\nQuery: '{q_in1}'")
    print(f"Refusal Triggered: {r_in1['refusal']}")
    print(f"Relevance Score (Max Sim): {r_in1['max_similarity']}")
    print(f"Sources Cited: {r_in1['sources']}")
    assert not r_in1["refusal"], "In-domain query incorrectly refused!"
    assert len(r_in1["sources"]) > 0, "No sources cited!"
    print("✅ In-domain test 1 passed.")

    # In-domain query 2
    q_in2 = "When should a patient seek emergency care for diabetic ketoacidosis?"
    r_in2 = rag.ask(q_in2)
    print(f"\nQuery: '{q_in2}'")
    print(f"Refusal Triggered: {r_in2['refusal']}")
    print(f"Sources Cited: {r_in2['sources']}")
    assert not r_in2["refusal"], "In-domain query 2 incorrectly refused!"
    print("✅ In-domain test 2 passed.")

    # Out-of-domain query (Alzheimer's)
    q_out1 = "What is the cure for Alzheimer's disease?"
    r_out1 = rag.ask(q_out1)
    print(f"\nQuery (Out of Scope): '{q_out1}'")
    print(f"Refusal Triggered: {r_out1['refusal']}")
    print(f"Max Sim Score: {r_out1['max_similarity']}")
    print(f"Answer: {r_out1['answer']}")
    assert r_out1["refusal"], "Out-of-domain query was not refused!"
    assert "I cannot answer this based on the provided healthcare documents." in r_out1["answer"]
    print("✅ Out-of-domain refusal guardrail test 1 passed.")

    # Out-of-domain query (Quantum Physics)
    q_out2 = "How do quantum computers factor large prime numbers using Shor's algorithm?"
    r_out2 = rag.ask(q_out2)
    print(f"\nQuery (Out of Scope): '{q_out2}'")
    print(f"Refusal Triggered: {r_out2['refusal']}")
    print(f"Max Sim Score: {r_out2['max_similarity']}")
    assert r_out2["refusal"], "Out-of-domain query was not refused!"
    print("✅ Out-of-domain refusal guardrail test 2 passed.")

    print("\n" + "=" * 70)
    print("🎉 ALL END-TO-END VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
