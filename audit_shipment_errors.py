import json

# Load the detailed results
with open('full_dataset_test_detailed_20250729_205854.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Find shipment misclassifications  
shipment_errors = []
for r in results:
    if (r['expected_category'] == 'الإرسالية' and 
        r['predicted_category'] != 'الإرسالية'):
        shipment_errors.append({
            'id': r['incident_id'],
            'description': r['description'][:200],
            'expected': r['expected_category'],
            'predicted': r['predicted_category'],
            'confidence': r['category_confidence']
        })

print("🔍 SHIPMENT MISCLASSIFICATION EXAMPLES:")
print("="*80)

correct_predictions = 0
for i, error in enumerate(shipment_errors[:10], 1):  # Show first 10
    print(f"\n{i}. ID {error['id']}:")
    print(f"   Text: {error['description']}...")
    print(f"   Expected: {error['expected']} → Predicted: {error['predicted']} ({error['confidence']:.2f})")
    
    # Analyze if the prediction seems more correct than the ground truth
    text_lower = error['description'].lower()
    
    # Check for payment context (which would make prediction wrong)
    payment_indicators = ["تم سداد", "سداد", "فاتورة مسددة", "لم ينعكس", "انتظار السداد"]
    has_payment_issue = any(indicator in text_lower for indicator in payment_indicators)
    
    # Check for certificate generation issues
    cert_issues = ["إصدار شهادة", "شهادة", "certificate", "error.*certificate", "خطأ.*شهادة"]
    import re
    has_cert_issue = any(re.search(pattern, text_lower) for pattern in cert_issues)
    
    if has_payment_issue:
        print(f"   ⚠️  CONTAINS PAYMENT CONTEXT - Ground truth might be wrong")
    elif error['predicted'] == 'الشهادات الصادرة من الهيئة' and has_cert_issue:
        print(f"   ✅ PREDICTION SEEMS CORRECT - Certificate/technical issue")
        correct_predictions += 1
    elif error['predicted'] == 'المدفوعات':
        print(f"   🔍 NEED TO CHECK - Predicted as payment issue")
    else:
        print(f"   ❓ UNCLEAR - Need manual review")

print(f"\n📊 Total shipment misclassifications: {len(shipment_errors)}")
print(f"📊 Predictions that seem more accurate than ground truth: {correct_predictions}")

# Let's also check the reverse - shipments classified as payments
reverse_errors = []
for r in results:
    if (r['expected_category'] == 'المدفوعات' and 
        r['predicted_category'] == 'الإرسالية'):
        reverse_errors.append({
            'id': r['incident_id'],
            'description': r['description'][:150],
            'expected': r['expected_category'],
            'predicted': r['predicted_category'],
        })

print(f"\n🔄 REVERSE: Payments classified as Shipments: {len(reverse_errors)}")
for error in reverse_errors[:5]:
    print(f"   ID {error['id']}: {error['description']}...")
