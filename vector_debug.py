#!/usr/bin/env python
"""
Debug script to check vector similarities for classification
"""
import asyncio
import logging
from src.agents.category_classifier_agent import CategoryClassifierAgent
from src.config.agent_config import BaseAgentConfig
from src.data.category_loader import CategoryLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_vector_search():
    """Debug vector search for problematic test cases"""
    
    print("=== VECTOR SEARCH DEBUG ===")
    
    # Initialize hierarchy
    loader = CategoryLoader()
    hierarchy = loader.load_from_csv('Category + SubCategory.csv')
    
    # Create config
    config = BaseAgentConfig(
        agent_name="category_classifier",
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=2000
    )
    
    # Create agent
    agent = await CategoryClassifierAgent.create(config, hierarchy)
    
    # Test cases that failed
    test_cases = [
        {
            "text": "تم تسجيل الدخول واستكمال البيانات، وتبين أن الشركة مسجلة.",
            "expected": "التسجيل",
            "description": "Registration issue"
        },
        {
            "text": "تم سداد فاتورة الشهادة الإرسالية ولكن الشهادة لم تظهر. حالة الطلب تظهر بأنها في انتظار السداد رغم أن الفاتورة مسددة.",
            "expected": "الإرسالية", 
            "description": "Shipment issue"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Text: {test['text']}")
        print(f"Expected: {test['expected']}")
        
        # Get vector similarities
        try:
            similar_categories = await agent._find_similar_categories(test['text'])
            print(f"\nTop 5 Vector Similarities:")
            for j, cat in enumerate(similar_categories[:5], 1):
                marker = "✅" if cat['name'] == test['expected'] else "❌"
                print(f"  {j}. {cat['name']} (score: {cat['similarity_score']:.3f}) {marker}")
                
            # Check if expected category is in top results
            expected_rank = None
            for j, cat in enumerate(similar_categories, 1):
                if cat['name'] == test['expected']:
                    expected_rank = j
                    break
                    
            if expected_rank:
                print(f"Expected category '{test['expected']}' is rank #{expected_rank}")
            else:
                print(f"⚠️  Expected category '{test['expected']}' not found in results!")
                
        except Exception as e:
            print(f"❌ Error getting similarities: {e}")

if __name__ == "__main__":
    asyncio.run(debug_vector_search())
