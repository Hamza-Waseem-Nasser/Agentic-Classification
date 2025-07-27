"""
LANGSMITH INTEGRATION FOR ITSM CLASSIFICATION
============================================

This script demonstrates LangSmith integration for advanced debugging and tracing
of the ITSM classification pipeline. LangSmith provides excellent observability
for LLM applications.

Setup Instructions:
1. Install LangSmith: pip install langsmith
2. Set environment variables:
   - LANGCHAIN_TRACING_V2=true
   - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   - LANGCHAIN_API_KEY=your-api-key
   - LANGCHAIN_PROJECT=itsm-classification

Features:
- Complete pipeline tracing
- LLM call monitoring
- Performance analytics
- Error tracking
- Chain of thought visualization
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"  # Uncomment and add your key
os.environ["LANGCHAIN_PROJECT"] = "itsm-classification-demo"

from langsmith import traceable
from main import initialize_classification_system, classify_ticket
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class LangSmithDemo:
    """Demo class with LangSmith tracing integration"""
    
    def __init__(self):
        self.pipeline = None
    
    @traceable(name="initialize_itsm_system")
    async def initialize_system(self):
        """Initialize the ITSM system with LangSmith tracing"""
        logger.info("Initializing ITSM Classification System with LangSmith tracing")
        self.pipeline = await initialize_classification_system(strict_mode=True)
        return self.pipeline
    
    @traceable(name="classify_itsm_ticket")
    async def classify_with_tracing(self, ticket_text: str, ticket_id: str) -> Dict[str, Any]:
        """Classify ticket with full LangSmith tracing"""
        
        # This will create a trace in LangSmith showing the complete pipeline
        result = await classify_ticket(self.pipeline, ticket_text, ticket_id)
        
        # Add custom metadata to the trace
        trace_metadata = {
            "ticket_id": ticket_id,
            "input_length": len(ticket_text),
            "processing_time": result.get('processing', {}).get('processing_time', 0),
            "category": result.get('classification', {}).get('category'),
            "subcategory": result.get('classification', {}).get('subcategory'),
            "category_confidence": result.get('classification', {}).get('category_confidence'),
            "subcategory_confidence": result.get('classification', {}).get('subcategory_confidence'),
            "success": result.get('success', False)
        }
        
        return {**result, "trace_metadata": trace_metadata}
    
    @traceable(name="demo_run")
    async def run_langsmith_demo(self):
        """Run demo with LangSmith integration"""
        
        print("🔍 LangSmith ITSM Classification Demo")
        print("=" * 50)
        print("📊 This demo will create traces in LangSmith for:")
        print("   - System initialization")
        print("   - Individual ticket classifications")
        print("   - Agent interactions")
        print("   - LLM calls and responses")
        print("=" * 50)
        print()
        
        # Sample tickets from your data
        sample_tickets = [
            {
                "id": "175391",
                "text": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله",
                "expected": "التسجيل"
            },
            {
                "id": "175395", 
                "text": "تم سداد فاتورة شهادة ارسالية ولم تظهر الشهادة حالة الطلب بانتظار السداد",
                "expected": "الإرسالية"
            },
            {
                "id": "175480",
                "text": "عدم وصول رابط استعادة كلمة المرور",
                "expected": "تسجيل الدخول"
            }
        ]
        
        # Initialize system
        await self.initialize_system()
        
        # Process tickets
        results = []
        for ticket in sample_tickets:
            print(f"🎫 Processing ticket {ticket['id']}")
            print(f"📝 Text: {ticket['text'][:80]}...")
            
            result = await self.classify_with_tracing(ticket['text'], ticket['id'])
            results.append(result)
            
            predicted = result.get('classification', {}).get('category', 'Unknown')
            confidence = result.get('classification', {}).get('category_confidence', 0)
            
            print(f"🎯 Predicted: {predicted} (confidence: {confidence:.2f})")
            print(f"✅ Expected: {ticket['expected']}")
            print(f"✅ Match: {'Yes' if predicted == ticket['expected'] else 'No'}")
            print()
        
        print("🎉 Demo completed!")
        print("🔍 Check your LangSmith dashboard for detailed traces:")
        print("   https://smith.langchain.com/")
        print(f"   Project: {os.environ.get('LANGCHAIN_PROJECT', 'itsm-classification-demo')}")
        
        return results


def setup_langsmith_environment():
    """Setup LangSmith environment and provide instructions"""
    
    print("🔧 LangSmith Setup Instructions")
    print("=" * 40)
    print()
    print("1. Install LangSmith:")
    print("   pip install langsmith")
    print()
    print("2. Get your API key from https://smith.langchain.com/")
    print()
    print("3. Set environment variables:")
    print("   export LANGCHAIN_TRACING_V2=true")
    print("   export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com")
    print("   export LANGCHAIN_API_KEY=your-api-key")
    print("   export LANGCHAIN_PROJECT=itsm-classification")
    print()
    print("4. Run this script to see traces in LangSmith dashboard")
    print()
    
    # Check if LangSmith is properly configured
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("⚠️  Warning: LANGCHAIN_API_KEY not set!")
        print("   Traces will not be sent to LangSmith.")
        print("   Set your API key to enable full tracing.")
    else:
        print("✅ LangSmith configuration detected")
        print("   Traces will be sent to LangSmith dashboard")
    
    print()


async def main():
    """Main function for LangSmith demo"""
    
    setup_langsmith_environment()
    
    # Ask user if they want to continue
    response = input("Continue with demo? (y/n): ").lower().strip()
    if response != 'y':
        print("Demo cancelled.")
        return
    
    # Run demo
    demo = LangSmithDemo()
    await demo.run_langsmith_demo()


if __name__ == "__main__":
    asyncio.run(main())
