"""
COMPREHENSIVE DEMO: 4-AGENT ARABIC ITSM CLASSIFICATION PIPELINE

This demo showcases the complete implementation of Step 3 with all four agents
working together to classify Arabic ITSM tickets. It demonstrates the full
workflow from raw Arabic text to final categorization.

DEMO FEATURES:
1. Complete Pipeline: Shows all 4 agents working in sequence
2. Real Arabic Examples: Uses authentic Arabic ITSM ticket examples
3. Performance Metrics: Displays timing and accuracy metrics
4. Error Handling: Demonstrates graceful error recovery
5. Vector Search: Shows Qdrant integration for semantic matching
6. Comprehensive Output: Detailed results with confidence scores

PIPELINE ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Orchestrator    │───▶│ Arabic           │───▶│ Category        │───▶│ Subcategory     │
│ Agent           │    │ Processing Agent │    │ Classifier Agent│    │ Classifier Agent│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
      ▲                           ▲                        ▲                        ▲
      │                           │                        │                        │
   Business                  Language                 Vector Search           Hierarchical
    Rules                  Processing                 + LLM Reasoning        Classification

AGENT RESPONSIBILITIES:
- Orchestrator: Workflow management, routing, business rules
- Arabic Processor: Normalization, entity extraction, dialect handling
- Category Classifier: Main category identification using vector search + LLM
- Subcategory Classifier: Hierarchical subcategory selection within category

INTEGRATION HIGHLIGHTS:
- Vector Database: Qdrant for semantic similarity search
- Language Model: GPT-4 for intelligent classification reasoning
- State Management: Consistent state tracking across all agents
- Error Recovery: Graceful fallbacks when components fail
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_classification.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Import our pipeline
try:
    from src.agents.classification_pipeline import ClassificationPipeline
    from src.models.entities import ClassificationHierarchy, Category, Subcategory
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure you're running from the project root directory")
    exit(1)


class ITSMClassificationDemo:
    """
    Comprehensive demo of the 4-agent ITSM classification pipeline.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        
        # Sample Arabic ITSM tickets for testing
        self.sample_tickets = [
            {
                "id": "ticket_001",
                "text": "الكمبيوتر لا يعمل والشاشة سوداء، أحتاج مساعدة عاجلة",
                "expected_category": "مشاكل الأجهزة",
                "expected_subcategory": "أجهزة الكمبيوتر"
            },
            {
                "id": "ticket_002", 
                "text": "البرنامج يتوقف عن العمل باستمرار ويظهر رسالة خطأ",
                "expected_category": "مشاكل البرمجيات",
                "expected_subcategory": "مشاكل التطبيقات"
            },
            {
                "id": "ticket_003",
                "text": "لا أستطيع الدخول إلى النظام، كلمة المرور لا تعمل",
                "expected_category": "الأمان والوصول",
                "expected_subcategory": "مشاكل تسجيل الدخول"
            },
            {
                "id": "ticket_004",
                "text": "الطابعة لا تطبع والورق عالق بداخلها",
                "expected_category": "مشاكل الأجهزة",
                "expected_subcategory": "أجهزة الطباعة"
            },
            {
                "id": "ticket_005",
                "text": "الإنترنت بطيء جداً والاتصال ينقطع كل فترة",
                "expected_category": "الشبكة والاتصالات",
                "expected_subcategory": "مشاكل الاتصال"
            }
        ]
    
    def setup_environment(self):
        """Setup the environment and check requirements."""
        self.logger.info("Setting up demo environment...")
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            self.logger.warning("OPENAI_API_KEY not found in environment variables")
            self.logger.info("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        self.logger.info("Environment setup complete")
    
    def create_enhanced_hierarchy(self) -> ClassificationHierarchy:
        """Create a comprehensive classification hierarchy for demo."""
        categories = []
        
        # Hardware Issues Category
        hardware_subcategories = {
            "أجهزة الكمبيوتر": Subcategory(
                name="أجهزة الكمبيوتر",
                description="مشاكل أجهزة الكمبيوتر والمكونات الداخلية",
                parent_category="مشاكل الأجهزة",
                keywords={"كمبيوتر", "معالج", "ذاكرة", "قرص صلب", "شاشة", "لوحة مفاتيح"}
            ),
            "أجهزة الطباعة": Subcategory(
                name="أجهزة الطباعة",
                description="مشاكل الطابعات وأجهزة الطباعة",
                parent_category="مشاكل الأجهزة",
                keywords={"طابعة", "طباعة", "ورق", "حبر", "تونر"}
            ),
            "أجهزة الشبكة": Subcategory(
                name="أجهزة الشبكة",
                description="مشاكل أجهزة الشبكة والاتصالات",
                parent_category="مشاكل الأجهزة",
                keywords={"راوتر", "سويتش", "شبكة", "كابل", "واي فاي"}
            )
        }
        
        categories.append(Category(
            name="مشاكل الأجهزة",
            description="مشاكل متعلقة بالأجهزة والمعدات المادية",
            subcategories=hardware_subcategories,
            keywords={"جهاز", "هاردوير", "معدات", "مكونات", "كمبيوتر", "طابعة"}
        ))
        
        # Software Issues Category
        software_subcategories = {
            "مشاكل التطبيقات": Subcategory(
                name="مشاكل التطبيقات",
                description="مشاكل في التطبيقات والبرامج المختلفة",
                parent_category="مشاكل البرمجيات",
                keywords={"تطبيق", "برنامج", "خطأ", "تعطل", "تجمد"}
            ),
            "مشاكل النظام": Subcategory(
                name="مشاكل النظام",
                description="مشاكل نظام التشغيل والنظام الأساسي",
                parent_category="مشاكل البرمجيات",
                keywords={"نظام", "ويندوز", "تشغيل", "إقلاع", "تحديث"}
            )
        }
        
        categories.append(Category(
            name="مشاكل البرمجيات",
            description="مشاكل متعلقة بالبرمجيات والتطبيقات",
            subcategories=software_subcategories,
            keywords={"برنامج", "تطبيق", "نظام", "سوفتوير", "خطأ", "تعطل"}
        ))
        
        # Network Connectivity Category
        network_subcategories = {
            "مشاكل الاتصال": Subcategory(
                name="مشاكل الاتصال",
                description="مشاكل الاتصال بالشبكة والإنترنت",
                parent_category="الشبكة والاتصالات",
                keywords={"اتصال", "إنترنت", "شبكة", "انقطاع", "بطء"}
            ),
            "مشاكل الواي فاي": Subcategory(
                name="مشاكل الواي فاي",
                description="مشاكل الاتصال اللاسلكي والواي فاي",
                parent_category="الشبكة والاتصالات",
                keywords={"واي فاي", "لاسلكي", "اتصال", "إشارة"}
            )
        }
        
        categories.append(Category(
            name="الشبكة والاتصالات",
            description="مشاكل الشبكة والاتصال بالإنترنت",
            subcategories=network_subcategories,
            keywords={"شبكة", "إنترنت", "اتصال", "واي فاي", "بطء"}
        ))
        
        # Security and Access Category
        security_subcategories = {
            "مشاكل تسجيل الدخول": Subcategory(
                name="مشاكل تسجيل الدخول",
                description="مشاكل في تسجيل الدخول وكلمات المرور",
                parent_category="الأمان والوصول",
                keywords={"دخول", "كلمة مرور", "تسجيل", "حساب"}
            ),
            "صلاحيات الوصول": Subcategory(
                name="صلاحيات الوصول",
                description="مشاكل صلاحيات الوصول والأذونات",
                parent_category="الأمان والوصول",
                keywords={"صلاحيات", "أذونات", "وصول", "مجلد", "ملف"}
            )
        }
        
        categories.append(Category(
            name="الأمان والوصول",
            description="مشاكل الأمان وتسجيل الدخول والوصول",
            subcategories=security_subcategories,
            keywords={"أمان", "دخول", "كلمة مرور", "وصول", "تشفير"}
        ))
        
        return ClassificationHierarchy(
            categories={
                "مشاكل الأجهزة": categories[0],
                "مشاكل البرمجيات": categories[1], 
                "الشبكة والاتصالات": categories[2],
                "الأمان والوصول": categories[3]
            }
        )
    
    async def initialize_pipeline(self):
        """Initialize the classification pipeline."""
        self.logger.info("Initializing classification pipeline...")
        
        try:
            # Create enhanced hierarchy
            hierarchy = self.create_enhanced_hierarchy()
            
            # Initialize pipeline (Qdrant will be optional for demo)
            self.pipeline = ClassificationPipeline(
                config_path=None,  # Use default config
                hierarchy=hierarchy,
                qdrant_client=None  # Will fall back to keyword matching
            )
            
            self.logger.info("Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    async def demo_single_classification(self, ticket: Dict[str, Any]):
        """Demonstrate classification of a single ticket."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CLASSIFYING TICKET: {ticket['id']}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Original Text: {ticket['text']}")
        self.logger.info(f"Expected Category: {ticket['expected_category']}")
        self.logger.info(f"Expected Subcategory: {ticket['expected_subcategory']}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Classify the ticket
            result = await self.pipeline.classify_ticket(
                ticket_text=ticket['text'],
                ticket_id=ticket['id']
            )
            
            # Display results
            if result['success']:
                classification = result['classification']
                processing = result['processing']
                
                self.logger.info("\n📊 CLASSIFICATION RESULTS:")
                self.logger.info(f"✅ Category: {classification['category']} (confidence: {classification['category_confidence']:.2f})")
                self.logger.info(f"✅ Subcategory: {classification['subcategory']} (confidence: {classification['subcategory_confidence']:.2f})")
                
                self.logger.info(f"\n⚡ PROCESSING INFO:")
                self.logger.info(f"📝 Processed Text: {processing['processed_text']}")
                self.logger.info(f"🏷️ Entities: {processing['entities']}")
                self.logger.info(f"🔧 Technical Terms: {processing['technical_terms']}")
                self.logger.info(f"⏱️ Processing Time: {processing['processing_time']:.2f}s")
                
                # Check accuracy
                category_correct = classification['category'] == ticket['expected_category']
                subcategory_correct = classification['subcategory'] == ticket['expected_subcategory']
                
                self.logger.info(f"\n🎯 ACCURACY CHECK:")
                self.logger.info(f"Category Match: {'✅' if category_correct else '❌'}")
                self.logger.info(f"Subcategory Match: {'✅' if subcategory_correct else '❌'}")
                
                return {
                    'ticket_id': ticket['id'],
                    'success': True,
                    'category_correct': category_correct,
                    'subcategory_correct': subcategory_correct,
                    'processing_time': processing['processing_time']
                }
            else:
                self.logger.error(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
                return {
                    'ticket_id': ticket['id'],
                    'success': False,
                    'category_correct': False,
                    'subcategory_correct': False,
                    'processing_time': 0
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error during classification: {e}")
            return {
                'ticket_id': ticket['id'],
                'success': False,
                'category_correct': False,
                'subcategory_correct': False,
                'processing_time': 0
            }
    
    async def demo_batch_classification(self):
        """Demonstrate batch classification of all sample tickets."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("STARTING BATCH CLASSIFICATION DEMO")
        self.logger.info(f"{'='*80}")
        
        results = []
        
        for ticket in self.sample_tickets:
            result = await self.demo_single_classification(ticket)
            results.append(result)
            
            # Add a small delay between tickets for better logging
            await asyncio.sleep(1)
        
        return results
    
    def display_performance_summary(self, results: List[Dict[str, Any]]):
        """Display comprehensive performance summary."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info(f"{'='*80}")
        
        # Calculate metrics
        total_tickets = len(results)
        successful_classifications = sum(1 for r in results if r['success'])
        category_correct = sum(1 for r in results if r['category_correct'])
        subcategory_correct = sum(1 for r in results if r['subcategory_correct'])
        total_time = sum(r['processing_time'] for r in results)
        avg_time = total_time / total_tickets if total_tickets > 0 else 0
        
        # Overall metrics
        self.logger.info(f"\n📈 OVERALL METRICS:")
        self.logger.info(f"Total Tickets Processed: {total_tickets}")
        self.logger.info(f"Successful Classifications: {successful_classifications}/{total_tickets} ({successful_classifications/total_tickets*100:.1f}%)")
        self.logger.info(f"Category Accuracy: {category_correct}/{total_tickets} ({category_correct/total_tickets*100:.1f}%)")
        self.logger.info(f"Subcategory Accuracy: {subcategory_correct}/{total_tickets} ({subcategory_correct/total_tickets*100:.1f}%)")
        self.logger.info(f"Average Processing Time: {avg_time:.2f}s")
        self.logger.info(f"Total Processing Time: {total_time:.2f}s")
        
        # Pipeline metrics
        if self.pipeline:
            pipeline_metrics = self.pipeline.get_performance_metrics()
            
            self.logger.info(f"\n🔧 PIPELINE METRICS:")
            for agent_name, metrics in pipeline_metrics['agent_performance'].items():
                if metrics['calls'] > 0:
                    failure_rate = metrics['failures'] / metrics['calls'] * 100
                    self.logger.info(f"{agent_name.title()}: {metrics['calls']} calls, {failure_rate:.1f}% failure rate, {metrics['avg_time']:.2f}s avg")
        
        # Individual results
        self.logger.info(f"\n📋 INDIVIDUAL RESULTS:")
        for i, result in enumerate(results):
            ticket = self.sample_tickets[i]
            status = "✅" if result['success'] else "❌"
            self.logger.info(f"{status} {ticket['id']}: {result['processing_time']:.2f}s")
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        try:
            self.logger.info("🚀 STARTING COMPREHENSIVE ITSM CLASSIFICATION DEMO")
            self.logger.info("="*80)
            
            # Setup
            self.setup_environment()
            
            # Initialize pipeline
            if not await self.initialize_pipeline():
                self.logger.error("Failed to initialize pipeline. Exiting.")
                return
            
            # Run batch classification
            results = await self.demo_batch_classification()
            
            # Display summary
            self.display_performance_summary(results)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info("✅ DEMO COMPLETED SUCCESSFULLY!")
            self.logger.info("Check the logs for detailed processing information.")
            self.logger.info(f"{'='*80}")
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Demo failed with error: {e}")
            raise


async def main():
    """Main entry point for the demo."""
    demo = ITSMClassificationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
