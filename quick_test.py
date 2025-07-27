#!/usr/bin/env python
"""
Quick test of improved classification
"""
import asyncio
import logging
from main import initialize_classification_system

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_test():
    """Quick test of improved classification"""
    
    print("=== QUICK CLASSIFICATION TEST ===")
    
    # Initialize system
    pipeline = await initialize_classification_system(strict_mode=True)
    print("✅ System initialized")
    
    # Test case 1: Registration issue
    text1 = "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله"
    print(f"\nTesting: {text1}")
    
    result1 = await pipeline.classify_ticket("test1", text1)
    print(f"Result: {result1['classification']['category']} (confidence: {result1['classification']['category_confidence']:.2f})")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(quick_test())
