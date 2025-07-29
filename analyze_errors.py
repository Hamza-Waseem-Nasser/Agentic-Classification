import json
from collections import Counter

# Load the detailed results
with open('full_dataset_test_detailed_20250729_205854.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Get all misclassified cases
errors = [r for r in results if not r['overall_correct']]
category_errors = [r for r in results if not r['category_correct']]

print(f"📊 ERROR ANALYSIS")
print(f"Total cases: {len(results)}")
print(f"Correctly classified: {len([r for r in results if r['overall_correct']])}")
print(f"Misclassified: {len(errors)}")
print(f"Category errors: {len(category_errors)}")
print()

# Top category confusion patterns
category_confusions = []
for r in category_errors:
    confusion = f"{r['expected_category']} → {r['predicted_category']}"
    category_confusions.append(confusion)

confusion_counts = Counter(category_confusions)
print("🔥 TOP CATEGORY CONFUSIONS:")
for confusion, count in confusion_counts.most_common(15):
    print(f"   {confusion}: {count} cases")
print()

# High confidence errors
high_conf_errors = [r for r in errors if r['category_confidence'] >= 0.9]
print(f"⚠️ HIGH CONFIDENCE ERRORS ({len(high_conf_errors)} cases):")
for error in high_conf_errors[:10]:  # Show first 10
    confusion = f"{error['expected_category']} → {error['predicted_category']}"
    print(f"   ID {error['incident_id']}: {confusion} ({error['category_confidence']:.2f})")
    # Show a truncated description
    desc = error['description'][:100] + "..." if len(error['description']) > 100 else error['description']
    print(f"   Text: {desc}")
    print()

# Payment related errors (المدفوعات)
payment_errors = [r for r in category_errors if r['expected_category'] == 'المدفوعات']
print(f"💰 PAYMENT CLASSIFICATION ERRORS ({len(payment_errors)} cases):")
payment_confusions = Counter([r['predicted_category'] for r in payment_errors])
for category, count in payment_confusions.most_common(5):
    print(f"   Classified as {category}: {count} cases")
print()

# Shipment related errors (الإرسالية) 
shipment_errors = [r for r in category_errors if r['expected_category'] == 'الإرسالية']
print(f"📦 SHIPMENT CLASSIFICATION ERRORS ({len(shipment_errors)} cases):")
shipment_confusions = Counter([r['predicted_category'] for r in shipment_errors])
for category, count in shipment_confusions.most_common(5):
    print(f"   Classified as {category}: {count} cases")
print()

# Specific examples for key error patterns
print("📋 SPECIFIC ERROR EXAMPLES:")
print()

# Payment → Shipment errors
payment_to_shipment = [r for r in category_errors if r['expected_category'] == 'المدفوعات' and r['predicted_category'] == 'الإرسالية']
if payment_to_shipment:
    print(f"💰→📦 Payment classified as Shipment ({len(payment_to_shipment)} cases):")
    for error in payment_to_shipment[:3]:
        print(f"   ID {error['incident_id']}: {error['description'][:150]}...")
    print()

# Shipment → Certificates errors
shipment_to_cert = [r for r in category_errors if r['expected_category'] == 'الإرسالية' and r['predicted_category'] == 'الشهادات الصادرة من الهيئة']
if shipment_to_cert:
    print(f"📦→📜 Shipment classified as Certificates ({len(shipment_to_cert)} cases):")
    for error in shipment_to_cert[:3]:
        print(f"   ID {error['incident_id']}: {error['description'][:150]}...")
    print()
