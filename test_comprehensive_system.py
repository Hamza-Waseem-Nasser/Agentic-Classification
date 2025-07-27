"""
Comprehensive test suite for ITSM classification system
Based on the enhanced test suite provided in comprehensive-test-suite.py
"""

import asyncio
import json
from typing import List, Dict, Any
from main import initialize_classification_system, classify_ticket

# Use centralized logging (main.py already sets it up)
from src.utils.logging_config import get_logger
logger = get_logger(__name__)

async def run_comprehensive_tests():
    """Run comprehensive tests on the classification system"""
    
    # Initialize system
    pipeline = await initialize_classification_system(strict_mode=True)
    
    # Extended test cases covering edge cases
    test_cases = [
        # Original test cases
        {
            "id": "175391",
            "text": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله (AutoClosed)",
            "expected_category": "التسجيل",
            "expected_subcategory": "التحقق من السجل التجاري",
            "test_type": "system_tag_removal"
        },
        {
            "id": "175395",
            "text": "تم سداد فاتورة شهادة ارسالية ولم تظهر الشهادة حالة الطلب بانتظار السداد مع العلم بأن الفاتورة مسدده  نرجو حل المشكلة (AutoClosed)",
            "expected_category": "الإرسالية",
            "expected_subcategory": "حالة الطلب في النظام",
            "test_type": "complex_classification"
        },
        
        # Edge cases
        {
            "id": "edge_001",
            "text": "مشكلة",  # Very short text
            "expected_category": "أخرى",
            "test_type": "short_text"
        },
        {
            "id": "edge_002", 
            "text": "I cannot login to the system نسيت كلمة المرور",  # Mixed language
            "expected_category": "تسجيل الدخول",
            "test_type": "mixed_language"
        },
        {
            "id": "edge_003",
            "text": "تم دفع مبلغ 500 ريال للشهادة ولكن الشهادة لم تصدر",  # Payment mentioned but shipment issue
            "expected_category": "الإرسالية",
            "test_type": "misleading_keywords"
        },
        {
            "id": "edge_004",
            "text": "ERROR CODE: SYS-001 عند محاولة إضافة منتج جديد",  # Technical error
            "expected_category": "إضافة المنتجات",
            "test_type": "technical_error"
        },
        {
            "id": "edge_005",
            "text": "تسجيل",  # Partial category name
            "expected_category": "التسجيل",
            "test_type": "partial_match"
        },
        {
            "id": "edge_006",
            "text": "الرجاء المساعدة في حل المشكلة بسرعة عاجل جداً",  # No clear category
            "expected_category": "أخرى",
            "test_type": "ambiguous"
        }
    ]
    
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE CLASSIFICATION SYSTEM TEST")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\nTest ID: {test_case['id']} ({test_case['test_type']})")
        print(f"Input: {test_case['text'][:50]}...")
        
        try:
            result = await classify_ticket(pipeline, test_case['text'], test_case['id'])
            
            # Extract results
            category = result.get('classification', {}).get('category')
            subcategory = result.get('classification', {}).get('subcategory')
            confidence = result.get('classification', {}).get('category_confidence', 0)
            processed_text = result.get('processing', {}).get('processed_text', '')
            
            # Check results
            category_match = category == test_case['expected_category']
            has_autoclosed = any(tag in processed_text.lower() for tag in ['autoclosed', 'مغلق تلقائ'])
            
            # Store results
            test_result = {
                'test_id': test_case['id'],
                'test_type': test_case['test_type'],
                'category_match': category_match,
                'predicted_category': category,
                'expected_category': test_case['expected_category'],
                'confidence': confidence,
                'autoclosed_removed': not has_autoclosed,
                'processing_time': result.get('processing', {}).get('processing_time', 0)
            }
            
            results.append(test_result)
            
            # Track accuracy if pipeline supports it
            if hasattr(pipeline, 'track_classification_accuracy'):
                pipeline.track_classification_accuracy(
                    test_case['expected_category'], 
                    category, 
                    confidence
                )
            
            # Print results
            status = "✅" if category_match else "❌"
            print(f"{status} Category: {category} (expected: {test_case['expected_category']})")
            print(f"   Confidence: {confidence:.2f}")
            
            if test_case['test_type'] == 'system_tag_removal':
                tag_status = "✅" if not has_autoclosed else "❌"
                print(f"{tag_status} AutoClosed removed: {not has_autoclosed}")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            results.append({
                'test_id': test_case['id'],
                'test_type': test_case['test_type'],
                'error': str(e),
                'category_match': False
            })
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    successful = sum(1 for r in results if r.get('category_match', False))
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful}/{total_tests} ({successful/total_tests*100:.1f}%)")
    
    # Group by test type
    by_type = {}
    for result in results:
        test_type = result['test_type']
        if test_type not in by_type:
            by_type[test_type] = {'total': 0, 'successful': 0}
        by_type[test_type]['total'] += 1
        if result.get('category_match', False):
            by_type[test_type]['successful'] += 1
    
    print("\nResults by Test Type:")
    for test_type, stats in by_type.items():
        success_rate = stats['successful'] / stats['total'] * 100
        print(f"  {test_type}: {stats['successful']}/{stats['total']} ({success_rate:.0f}%)")
    
    # Confidence distribution
    confidences = [r['confidence'] for r in results if 'confidence' in r]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"\nAverage Confidence: {avg_confidence:.2f}")
        print(f"Min Confidence: {min(confidences):.2f}")
        print(f"Max Confidence: {max(confidences):.2f}")
    
    # Processing time
    times = [r['processing_time'] for r in results if 'processing_time' in r]
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage Processing Time: {avg_time:.2f}s")
    
    # Pipeline accuracy stats (if available)
    if hasattr(pipeline, 'accuracy_stats'):
        print(f"\nPipeline Accuracy Stats:")
        stats = pipeline.accuracy_stats
        if stats['total'] > 0:
            overall_acc = stats['correct'] / stats['total'] * 100
            print(f"  Overall Accuracy: {overall_acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Save detailed results
    with open('test_results_comprehensive.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: test_results_comprehensive.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
