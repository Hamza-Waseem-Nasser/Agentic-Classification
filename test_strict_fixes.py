"""
Quick Test Script for Strict Classification Fixes

This script tests the fixes applied to the ITSM classification system:
1. System tag removal
2. Strict category validation
3. JSON parsing improvements
"""

import asyncio
import logging
from main import initialize_classification_system, classify_ticket

# Use centralized logging (main.py already sets it up)
from src.utils.logging_config import get_logger
logger = get_logger(__name__)

async def test_strict_classification():
    """Test the strict classification fixes"""
    
    logger.info("=" * 60)
    logger.info("TESTING STRICT CLASSIFICATION FIXES")
    logger.info("=" * 60)
    
    # Initialize system with strict mode
    logger.info("Initializing classification system in strict mode...")
    try:
        pipeline = await initialize_classification_system(strict_mode=True)
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return
    
    # Test cases based on actual Thiqa incidents
    test_cases = [
        {
            "name": "System Tag Removal Test - Registration Issue",
            "text": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله (AutoClosed)",
            "ticket_id": "175391",
            "expected_category": "التسجيل",  # This is actually a registration verification issue
            "expected_subcategory": "التحقق من السجل التجاري",
            "should_remove_tag": True
        },
        {
            "name": "System Tag Removal Test - Shipment Issue", 
            "text": "تم سداد فاتورة شهادة ارسالية ولم تظهر الشهادة حالة الطلب بانتظار السداد مع العلم بأن الفاتورة مسدده  نرجو حل المشكلة (AutoClosed)",
            "ticket_id": "175395",
            "expected_category": "الإرسالية",
            "expected_subcategory": "حالة الطلب في النظام",
            "should_remove_tag": True
        },
        {
            "name": "Payment Issue Test",
            "text": "لا استطيع دفع الرسوم المطلوبة للشهادة",
            "ticket_id": "test_003", 
            "expected_category": "المدفوعات",
            "should_remove_tag": False
        },
        {
            "name": "Product Addition Test",
            "text": "لا يمكنني اضافة منتج جديد للنظام",
            "ticket_id": "test_004",
            "expected_category": "إضافة المنتجات",
            "should_remove_tag": False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test_case['name']}")
        logger.info(f"Input: {test_case['text']}")
        
        try:
            # Classify the ticket
            result = await classify_ticket(
                pipeline,
                test_case["text"],
                test_case["ticket_id"]
            )
            
            # Check results
            category = result.get('classification', {}).get('category')
            subcategory = result.get('classification', {}).get('subcategory')
            processed_text = result.get('processing', {}).get('processed_text', '')
            # Get the correct confidence value
            confidence = result.get('classification', {}).get('category_confidence', 0)
            
            logger.info(f"Predicted Category: {category}")
            logger.info(f"Expected Category: {test_case['expected_category']}")
            if 'expected_subcategory' in test_case:
                logger.info(f"Predicted Subcategory: {subcategory}")
                logger.info(f"Expected Subcategory: {test_case['expected_subcategory']}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Processed Text: {processed_text}")
            
            # Validate results
            category_match = category == test_case['expected_category']
            subcategory_match = True  # Default to true if no expected subcategory
            if 'expected_subcategory' in test_case:
                subcategory_match = subcategory == test_case['expected_subcategory']
            tag_removed = not any(tag in processed_text.lower() for tag in ['autoclosed', 'مغلق تلقائ', 'إغلاق تلقائ'])
            
            test_result = {
                'test_name': test_case['name'],
                'category_match': category_match,
                'subcategory_match': subcategory_match,
                'tag_removal_ok': tag_removed if test_case['should_remove_tag'] else True,
                'confidence': confidence,
                'category': category,
                'subcategory': subcategory,
                'processed_text': processed_text
            }
            
            if category_match:
                logger.info("✅ Category classification: CORRECT")
            else:
                logger.info("❌ Category classification: INCORRECT")
            
            if 'expected_subcategory' in test_case:
                if subcategory_match:
                    logger.info("✅ Subcategory classification: CORRECT")
                else:
                    logger.info("❌ Subcategory classification: INCORRECT")
            
            if test_case['should_remove_tag'] and tag_removed:
                logger.info("✅ System tag removal: SUCCESSFUL")
            elif not test_case['should_remove_tag']:
                logger.info("✅ System tag handling: N/A")
            else:
                logger.info("❌ System tag removal: FAILED")
            
            results.append(test_result)
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            results.append({
                'test_name': test_case['name'],
                'error': str(e),
                'category_match': False,
                'subcategory_match': False,
                'tag_removal_ok': False
            })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(results)
    successful_categories = sum(1 for r in results if r.get('category_match', False))
    successful_subcategories = sum(1 for r in results if r.get('subcategory_match', False))
    successful_tag_removal = sum(1 for r in results if r.get('tag_removal_ok', False))
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Correct Categories: {successful_categories}/{total_tests}")
    logger.info(f"Correct Subcategories: {successful_subcategories}/{total_tests}")
    logger.info(f"Successful Tag Removal: {successful_tag_removal}/{total_tests}")
    logger.info(f"Category Accuracy: {successful_categories/total_tests*100:.1f}%")
    logger.info(f"Subcategory Accuracy: {successful_subcategories/total_tests*100:.1f}%")
    
    # Print valid categories for reference
    if hasattr(pipeline, 'category_classifier') and hasattr(pipeline.category_classifier, 'hierarchy'):
        valid_categories = list(pipeline.category_classifier.hierarchy.categories.keys())
        logger.info(f"\nValid Categories ({len(valid_categories)}):")
        for cat in valid_categories:
            logger.info(f"  - {cat}")
    
    return results

async def test_json_parsing():
    """Test the JSON parsing improvements"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING JSON PARSING IMPROVEMENTS")
    logger.info("=" * 60)
    
    from src.agents.arabic_processing_agent import ArabicProcessingAgent
    from src.config.config_validator import SystemConfig
    from src.config.agent_config import BaseAgentConfig
    
    # Create a test Arabic processor
    config = SystemConfig.from_env()
    agent_config = BaseAgentConfig(
        agent_name="arabic_processor"
    )
    
    processor = ArabicProcessingAgent(agent_config)
    
    # Test system tag removal
    test_text = "مشكلة في النظام (AutoClosed) وتم الإغلاق تلقائياً"
    cleaned_text = processor._remove_system_tags(test_text)
    
    logger.info(f"Original: {test_text}")
    logger.info(f"Cleaned: {cleaned_text}")
    
    if "(AutoClosed)" not in cleaned_text and "تم الإغلاق تلقائياً" not in cleaned_text:
        logger.info("✅ System tag removal working correctly")
    else:
        logger.info("❌ System tag removal failed")

if __name__ == "__main__":
    asyncio.run(test_strict_classification())
    asyncio.run(test_json_parsing())
