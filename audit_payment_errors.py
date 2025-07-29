import json

# Load the detailed results
with open('full_dataset_test_detailed_20250729_205854.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Find payment misclassifications specifically
payment_errors = []
for r in results:
    if (r['expected_category'] == 'المدفوعات' and 
        r['predicted_category'] != 'المدفوعات'):
        payment_errors.append({
            'id': r['incident_id'],
            'description': r['description'][:200],
            'expected': r['expected_category'],
            'predicted': r['predicted_category'],
            'confidence': r['category_confidence']
        })

print("🔍 PAYMENT MISCLASSIFICATION EXAMPLES:")
print("="*80)

for i, error in enumerate(payment_errors[:10], 1):  # Show first 10
    print(f"\n{i}. ID {error['id']}:")
    print(f"   Text: {error['description']}...")
    print(f"   Expected: {error['expected']} → Predicted: {error['predicted']} ({error['confidence']:.2f})")
    
    # Check if our payment detection would catch this
    text_lower = error['description'].lower()
    
    # Check payment patterns
    payment_patterns = [
        "تم سداد فاتورة.*لم تظهر",
        "سداد.*فاتورة.*لم.*ينعكس", 
        "بعد سداد.*لا تنعكس",
        "فاتورة.*مسددة.*لم تظهر",
        "لم يتم عكس السداد"
    ]
    
    import re
    detected = False
    for pattern in payment_patterns:
        if re.search(pattern, text_lower):
            detected = True
            print(f"   ✅ NEW LOGIC WOULD DETECT: Pattern '{pattern}' matches")
            break
    
    if not detected:
        payment_keywords = ["سداد", "فاتورة", "دفع", "مسدد", "مدفوع"]
        problem_keywords = ["لم تظهر", "لم ينعكس", "معلق", "انتظار السداد", "مشكلة"]
        
        has_payment = any(keyword in text_lower for keyword in payment_keywords)
        has_problem = any(keyword in text_lower for keyword in problem_keywords)
        
        if has_payment and has_problem:
            print(f"   🟡 NEW LOGIC WOULD DETECT: General payment + problem context")
        else:
            print(f"   ❌ NEW LOGIC MIGHT MISS: Needs additional pattern")
            print(f"      Payment keywords: {[k for k in payment_keywords if k in text_lower]}")
            print(f"      Problem keywords: {[k for k in problem_keywords if k in text_lower]}")

print(f"\n📊 Total payment misclassifications: {len(payment_errors)}")
