"""
Test script to run the classification pipeline with the two sample tickets
"""

import asyncio
import logging
import sys
import os
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import initialize_classification_system, classify_ticket

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test tickets
test_tickets = [
    {
        "id": "175391",
        "description": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله (AutoClosed)",
        "expected_category": "التسجيل",
        "expected_subcategory": "التحقق من السجل التجاري"
    },
    {
        "id": "175395",
        "description": "تم سداد فاتورة شهادة ارسالية ولم تظهر الشهادة حالة الطلب بانتظار السداد مع العلم بأن الفاتورة مسدده  نرجو حل المشكلة (AutoClosed)",
        "expected_category": "الإرسالية",
        "expected_subcategory": "حالة الطلب في النظام"
    }
]

async def test_classification_pipeline():
    """Test the classification pipeline with sample tickets"""
    
    try:
        logger.info("Starting classification pipeline test...")
        
        # Initialize the classification system
        logger.info("Initializing classification system...")
        pipeline = await initialize_classification_system()
        
        logger.info("Classification system initialized successfully!")
        
        # Test each ticket
        results = []
        for ticket in test_tickets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Ticket ID: {ticket['id']}")
            logger.info(f"Description: {ticket['description']}")
            logger.info(f"Expected Category: {ticket['expected_category']}")
            logger.info(f"Expected Subcategory: {ticket['expected_subcategory']}")
            logger.info(f"{'='*60}")
            
            # Classify the ticket
            result = await classify_ticket(
                pipeline=pipeline,
                ticket_text=ticket['description'],
                ticket_id=ticket['id']
            )
            
            # Extract classification results from the nested structure
            classification = result.get('classification', {})
            category = classification.get('category')
            subcategory = classification.get('subcategory')
            
            # Add expected values for comparison
            result['expected_category'] = ticket['expected_category']
            result['expected_subcategory'] = ticket['expected_subcategory']
            
            # Store extracted values for easier access
            result['category'] = category
            result['subcategory'] = subcategory
            result['confidence'] = classification.get('category_confidence', 0.0)
            
            # Calculate processing time in ms if available
            processing_info = result.get('processing', {})
            if 'processing_time' in processing_info:
                result['processing_time_ms'] = processing_info['processing_time'] * 1000
            
            # Check accuracy
            category_match = category == ticket['expected_category']
            subcategory_match = subcategory == ticket['expected_subcategory']
            
            result['category_match'] = category_match
            result['subcategory_match'] = subcategory_match
            result['overall_match'] = category_match and subcategory_match
            
            results.append(result)
            
            # Print results
            logger.info(f"\nCLASSIFICATION RESULTS:")
            logger.info(f"  Predicted Category: {result.get('category', 'N/A')}")
            logger.info(f"  Expected Category:  {ticket['expected_category']}")
            logger.info(f"  Category Match:     {'✅' if category_match else '❌'}")
            logger.info(f"  ")
            logger.info(f"  Predicted Subcategory: {result.get('subcategory', 'N/A')}")
            logger.info(f"  Expected Subcategory:  {ticket['expected_subcategory']}")
            logger.info(f"  Subcategory Match:     {'✅' if subcategory_match else '❌'}")
            logger.info(f"  ")
            logger.info(f"  Overall Accuracy:  {'✅' if result['overall_match'] else '❌'}")
            logger.info(f"  Confidence Score:  {result.get('confidence', 'N/A')}")
            logger.info(f"  Processing Time:   {result.get('processing_time_ms', 'N/A')} ms")
            logger.info(f"  Status:           {result.get('status', 'N/A')}")
            
            if 'error' in result:
                logger.error(f"  Error: {result['error']}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        total_tickets = len(results)
        successful_tickets = len([r for r in results if 'error' not in r])
        category_matches = len([r for r in results if r.get('category_match', False)])
        subcategory_matches = len([r for r in results if r.get('subcategory_match', False)])
        overall_matches = len([r for r in results if r.get('overall_match', False)])
        
        logger.info(f"Total Tickets Tested:     {total_tickets}")
        logger.info(f"Successfully Processed:   {successful_tickets}")
        logger.info(f"Category Accuracy:        {category_matches}/{total_tickets} ({category_matches/total_tickets*100:.1f}%)")
        logger.info(f"Subcategory Accuracy:     {subcategory_matches}/{total_tickets} ({subcategory_matches/total_tickets*100:.1f}%)")
        logger.info(f"Overall Accuracy:         {overall_matches}/{total_tickets} ({overall_matches/total_tickets*100:.1f}%)")
        
        # Save detailed results
        output_file = "test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if we have the necessary files
    required_files = [
        "Category + SubCategory.csv",
        "requirements.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        sys.exit(1)
    
    # Run the test
    results = asyncio.run(test_classification_pipeline())
    
    if results:
        logger.info("Pipeline test completed successfully!")
    else:
        logger.error("Pipeline test failed!")
        sys.exit(1)
