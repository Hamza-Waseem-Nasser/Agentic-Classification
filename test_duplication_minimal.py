"""
Minimal test to reproduce the exact duplication issue you reported
"""

import asyncio
from main import initialize_classification_system, classify_ticket

async def test_duplication_issue():
    """Test the exact scenario that was showing duplicates"""
    
    print("=== TESTING FOR DUPLICATION ISSUE ===")
    
    # Initialize system (this is where duplicates occurred)
    pipeline = await initialize_classification_system(strict_mode=True)
    
    # Test with one classification 
    test_text = "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله"
    print(f"\nTesting classification with: {test_text[:50]}...")
    
    result = await classify_ticket(pipeline, test_text, "test_duplication")
    
    category = result.get('classification', {}).get('category')
    print(f"Result: {category}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_duplication_issue())
