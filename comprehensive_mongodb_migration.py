"""
COMPREHENSIVE MONGODB MIGRATION PLAN
===================================

This script organizes and migrates ALL your ITSM classification data to MongoDB.
Your workspace contains a lot of valuable data that your CTO can use.

Data Categories Found:
1. Category Hierarchies (2 files)
2. Training Data (2 files) 
3. Enhanced Examples (2 files)
4. Accuracy Test Results (4 files)
5. System Configuration Data
6. Performance Reports
"""

import asyncio
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import os

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    PYMONGO_AVAILABLE = True
except ImportError:
    print("❌ PyMongo/Motor not installed. Install with: pip install motor pymongo")
    PYMONGO_AVAILABLE = False

class ComprehensiveMongoMigrator:
    def __init__(self, mongodb_url: str = "mongodb://localhost:27017", 
                 database_name: str = "itsm_classification"):
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.client = None
        self.db = None
        self.migration_summary = {
            "categories": 0,
            "training_examples": 0,
            "enhanced_examples": 0,
            "accuracy_results": 0,
            "system_prompts": 0,
            "performance_reports": 0
        }
    
    async def connect(self):
        """Connect to MongoDB"""
        if not PYMONGO_AVAILABLE:
            raise Exception("PyMongo/Motor not available")
        
        print(f"🔗 Connecting to MongoDB: {self.mongodb_url}")
        self.client = AsyncIOMotorClient(self.mongodb_url)
        self.db = self.client[self.database_name]
        
        # Test connection
        await self.client.admin.command('ping')
        print("✅ Connected to MongoDB")
    
    # 1. CATEGORY HIERARCHIES
    async def migrate_category_hierarchies(self):
        """Migrate category hierarchy files"""
        print("\n📂 1. MIGRATING CATEGORY HIERARCHIES")
        print("-" * 40)
        
        # Main category file
        await self._migrate_main_categories("Category + SubCategory.csv")
        
        # Additional category data from Thiqa
        await self._migrate_thiqa_categories("Thiqa_Incidents_Example.csv")
    
    async def _migrate_main_categories(self, filename):
        """Migrate main category hierarchy"""
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            return
        
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            categories = []
            
            for _, row in df.iterrows():
                category_doc = {
                    "category": str(row.get('Category', '')),
                    "category_description": str(row.get('Category_Description', '')),
                    "subcategory": str(row.get('SubCategory', '')),
                    "subcategory_description": str(row.get('SubCategory_Description', '')),
                    "source": "main_hierarchy",
                    "created_at": datetime.now(),
                    "active": True
                }
                categories.append(category_doc)
            
            if categories:
                await self.db.category_hierarchy.insert_many(categories)
                self.migration_summary["categories"] += len(categories)
                print(f"✅ Migrated {len(categories)} category definitions from {filename}")
            
        except Exception as e:
            print(f"❌ Failed to migrate {filename}: {e}")
    
    async def _migrate_thiqa_categories(self, filename):
        """Migrate Thiqa category data"""
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            return
        
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            categories = []
            
            for _, row in df.iterrows():
                category_doc = {
                    "incident_id": str(row.get('Incident', '')),
                    "description": str(row.get('Description', '')),
                    "subcategory_thiqah": str(row.get('Subcategory_Thiqah', '')),
                    "subcategory2_thiqah": str(row.get('Subcategory2_Thiqah', '')),
                    "source": "thiqa_data",
                    "created_at": datetime.now()
                }
                categories.append(category_doc)
            
            if categories:
                await self.db.thiqa_categories.insert_many(categories)
                self.migration_summary["categories"] += len(categories)
                print(f"✅ Migrated {len(categories)} Thiqa category mappings")
            
        except Exception as e:
            print(f"❌ Failed to migrate Thiqa categories: {e}")
    
    # 2. TRAINING EXAMPLES
    async def migrate_training_examples(self):
        """Migrate all training data"""
        print("\n📂 2. MIGRATING TRAINING EXAMPLES")
        print("-" * 40)
        
        # Main training data
        await self._migrate_main_training_data("Correct User Descrioptions.csv")
        
        # Enhanced training examples from Python file
        await self._migrate_enhanced_examples()
        
        # Error-based training examples
        await self._migrate_error_based_examples("training_examples_from_errors.json")
    
    async def _migrate_main_training_data(self, filename):
        """Migrate main training dataset"""
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            return
        
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            examples = []
            
            subcategory_col = 'Subcatgory' if 'Subcatgory' in df.columns else 'Subcategory'
            
            for _, row in df.iterrows():
                example_doc = {
                    "incident_id": str(row.get('Incident', '')),
                    "description": str(row.get('Description', '')),
                    "category": str(row.get('Category', '')),
                    "subcategory": str(row.get(subcategory_col, '')),
                    "source": "main_training_data",
                    "quality": "ground_truth",
                    "created_at": datetime.now(),
                    "usage": "training_and_testing"
                }
                examples.append(example_doc)
            
            if examples:
                await self.db.training_examples.insert_many(examples)
                self.migration_summary["training_examples"] += len(examples)
                print(f"✅ Migrated {len(examples)} main training examples")
            
        except Exception as e:
            print(f"❌ Failed to migrate main training data: {e}")
    
    async def _migrate_enhanced_examples(self):
        """Migrate enhanced examples from Python file"""
        try:
            # Import the enhanced examples
            import enhanced_training_examples
            
            enhanced_data = enhanced_training_examples.ENHANCED_CATEGORY_EXAMPLES
            examples = []
            
            for category, example_list in enhanced_data.items():
                for example_text in example_list:
                    example_doc = {
                        "description": example_text,
                        "category": category,
                        "source": "enhanced_examples",
                        "quality": "manually_curated",
                        "purpose": "embedding_training",
                        "created_at": datetime.now()
                    }
                    examples.append(example_doc)
            
            if examples:
                await self.db.enhanced_examples.insert_many(examples)
                self.migration_summary["enhanced_examples"] += len(examples)
                print(f"✅ Migrated {len(examples)} enhanced training examples")
            
        except Exception as e:
            print(f"❌ Failed to migrate enhanced examples: {e}")
    
    async def _migrate_error_based_examples(self, filename):
        """Migrate error-based training examples"""
        if not os.path.exists(filename):
            print(f"❌ {filename} not found")
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            error_patterns = data.get("training_examples_based_on_misclassifications", {})
            
            # Extract examples from each error pattern
            for pattern_name, pattern_data in error_patterns.items():
                if isinstance(pattern_data, dict) and "examples" in pattern_data:
                    for example in pattern_data["examples"]:
                        example_doc = {
                            "description": example.get("text", ""),
                            "category": pattern_data.get("category", ""),
                            "error_pattern": pattern_name,
                            "source": "error_analysis",
                            "quality": "derived_from_errors",
                            "purpose": "error_correction",
                            "created_at": datetime.now()
                        }
                        examples.append(example_doc)
            
            if examples:
                await self.db.error_based_examples.insert_many(examples)
                self.migration_summary["enhanced_examples"] += len(examples)
                print(f"✅ Migrated {len(examples)} error-based training examples")
            
        except Exception as e:
            print(f"❌ Failed to migrate error-based examples: {e}")
    
    # 3. ACCURACY TEST RESULTS
    async def migrate_accuracy_results(self):
        """Migrate accuracy test results and performance data"""
        print("\n📂 3. MIGRATING ACCURACY TEST RESULTS")
        print("-" * 40)
        
        # Find all accuracy test files
        accuracy_files = [
            "accuracy_test_detailed_20250729_164120.json",
            "accuracy_test_detailed_20250729_235847.json"
        ]
        
        for filename in accuracy_files:
            if os.path.exists(filename):
                await self._migrate_accuracy_file(filename)
    
    async def _migrate_accuracy_file(self, filename):
        """Migrate individual accuracy test file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store main test metadata and results
            test_doc = {
                "filename": filename,
                "test_metadata": data.get("test_metadata", {}),
                "accuracy_metrics": data.get("accuracy_metrics", {}),
                "error_analysis": data.get("error_analysis", {}),
                "improvement_suggestions": data.get("improvement_suggestions", []),
                "created_at": datetime.now(),
                "test_type": "accuracy_validation"
            }
            
            await self.db.accuracy_test_results.insert_one(test_doc)
            
            # Store individual test results for analysis
            detailed_results = data.get("detailed_results", [])
            if detailed_results:
                for result in detailed_results:
                    result["test_file"] = filename
                    result["created_at"] = datetime.now()
                
                await self.db.individual_test_results.insert_many(detailed_results)
            
            self.migration_summary["accuracy_results"] += 1
            print(f"✅ Migrated accuracy test: {filename} ({len(detailed_results)} individual results)")
            
        except Exception as e:
            print(f"❌ Failed to migrate {filename}: {e}")
    
    # 4. SYSTEM PROMPTS AND CONFIGURATION
    async def migrate_system_configuration(self):
        """Migrate system prompts and configuration"""
        print("\n📂 4. MIGRATING SYSTEM CONFIGURATION")
        print("-" * 40)
        
        # System prompts for each agent
        system_prompts = {
            "orchestrator": {
                "prompt": """أنت وكيل منسق لنظام تصنيف تذاكر الخدمات التقنية في هيئة المواصفات والمقاييس والجودة.
                مهمتك هي تحليل التذكرة الواردة وإعداد السياق للوكلاء المتخصصين لتصنيف دقيق.""",
                "version": "2.0",
                "agent_type": "orchestrator"
            },
            "arabic_processor": {
                "prompt": """أنت متخصص في معالجة النصوص العربية لتذاكر الخدمات التقنية.
                مهمتك هي تنظيف وتطبيع النص العربي واستخراج المصطلحات التقنية المتعلقة بمنصة سابر.""",
                "version": "2.0",
                "agent_type": "arabic_processor"
            },
            "category_classifier": {
                "prompt": """أنت متخصص في تصنيف تذاكر الخدمات التقنية إلى فئات رئيسية لمنصة سابر.
                استخدم النص المعالج لتحديد الفئة الأنسب من بين الفئات المتاحة مع درجة الثقة.""",
                "version": "2.0",
                "agent_type": "category_classifier"
            },
            "subcategory_classifier": {
                "prompt": """أنت متخصص في تصنيف تذاكر الخدمات التقنية إلى فئات فرعية لمنصة سابر.
                استخدم الفئة الرئيسية والنص المعالج لتحديد الفئة الفرعية الأنسب.""",
                "version": "2.0",
                "agent_type": "subcategory_classifier"
            }
        }
        
        for agent_type, config in system_prompts.items():
            prompt_doc = {
                "agent_type": agent_type,
                "prompt": config["prompt"],
                "version": config["version"],
                "created_at": datetime.now(),
                "active": True,
                "language": "arabic",
                "domain": "itsm_saber"
            }
            
            await self.db.system_prompts.replace_one(
                {"agent_type": agent_type, "version": config["version"]},
                prompt_doc,
                upsert=True
            )
        
        self.migration_summary["system_prompts"] = len(system_prompts)
        print(f"✅ Migrated {len(system_prompts)} system prompts")
    
    # 5. PERFORMANCE REPORTS
    async def migrate_performance_reports(self):
        """Migrate performance analysis reports"""
        print("\n📂 5. MIGRATING PERFORMANCE REPORTS")
        print("-" * 40)
        
        report_files = [
            "ACCURACY_IMPROVEMENT_REPORT.md",
            "FINAL_AUDIT_VALIDATION_REPORT.md",
            "AI_Architecture_Focused_Audit.md"
        ]
        
        for filename in report_files:
            if os.path.exists(filename):
                await self._migrate_report_file(filename)
    
    async def _migrate_report_file(self, filename):
        """Migrate individual report file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            report_doc = {
                "filename": filename,
                "content": content,
                "report_type": filename.replace('.md', '').lower(),
                "created_at": datetime.now(),
                "source": "system_analysis",
                "format": "markdown"
            }
            
            await self.db.performance_reports.insert_one(report_doc)
            self.migration_summary["performance_reports"] += 1
            print(f"✅ Migrated report: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to migrate {filename}: {e}")
    
    # 6. CREATE INDEXES FOR PERFORMANCE
    async def create_indexes(self):
        """Create database indexes for better performance"""
        print("\n🔧 CREATING DATABASE INDEXES")
        print("-" * 40)
        
        try:
            # Category hierarchy indexes
            await self.db.category_hierarchy.create_index("category")
            await self.db.category_hierarchy.create_index("subcategory")
            await self.db.category_hierarchy.create_index("source")
            
            # Training examples indexes
            await self.db.training_examples.create_index("category")
            await self.db.training_examples.create_index("subcategory")
            await self.db.training_examples.create_index("source")
            await self.db.training_examples.create_index("quality")
            
            # Enhanced examples indexes
            await self.db.enhanced_examples.create_index("category")
            await self.db.enhanced_examples.create_index("source")
            await self.db.enhanced_examples.create_index("purpose")
            
            # Accuracy results indexes
            await self.db.accuracy_test_results.create_index("test_metadata.timestamp")
            await self.db.individual_test_results.create_index("incident_id")
            await self.db.individual_test_results.create_index("success")
            
            # System prompts indexes
            await self.db.system_prompts.create_index("agent_type")
            await self.db.system_prompts.create_index("version")
            await self.db.system_prompts.create_index("active")
            
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create indexes: {e}")
    
    async def print_migration_summary(self):
        """Print migration summary"""
        print("\n🎉 MIGRATION COMPLETED!")
        print("=" * 50)
        print("📊 MIGRATION SUMMARY:")
        print(f"   📁 Categories: {self.migration_summary['categories']}")
        print(f"   📝 Training Examples: {self.migration_summary['training_examples']}")
        print(f"   ⭐ Enhanced Examples: {self.migration_summary['enhanced_examples']}")
        print(f"   📈 Accuracy Results: {self.migration_summary['accuracy_results']}")
        print(f"   🤖 System Prompts: {self.migration_summary['system_prompts']}")
        print(f"   📋 Performance Reports: {self.migration_summary['performance_reports']}")
        print()
        print("🗄️ MONGODB COLLECTIONS CREATED:")
        print("   • category_hierarchy - Category definitions")
        print("   • thiqa_categories - Thiqa mappings")
        print("   • training_examples - Main training data")
        print("   • enhanced_examples - Curated examples")
        print("   • error_based_examples - Error-derived examples")
        print("   • accuracy_test_results - Test summaries")
        print("   • individual_test_results - Detailed results")
        print("   • system_prompts - Agent prompts")
        print("   • performance_reports - Analysis reports")
        print()
        print("🚀 Your CTO now has ALL your data organized in MongoDB!")
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("🔄 Database connection closed")

async def main():
    """Main migration function"""
    print("🚀 COMPREHENSIVE ITSM CLASSIFICATION DATA MIGRATION")
    print("=" * 60)
    print("This will migrate ALL your data to MongoDB:")
    print("• Category hierarchies")
    print("• Training examples")  
    print("• Enhanced examples")
    print("• Accuracy test results")
    print("• System prompts")
    print("• Performance reports")
    print("=" * 60)
    
    # Get MongoDB connection details
    mongodb_url = input("Enter MongoDB URL (default: mongodb://localhost:27017): ").strip()
    if not mongodb_url:
        mongodb_url = "mongodb://localhost:27017"
    
    database_name = input("Enter database name (default: itsm_classification): ").strip()
    if not database_name:
        database_name = "itsm_classification"
    
    print(f"\n🎯 Target: {mongodb_url}/{database_name}")
    confirm = input("Continue with migration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Migration cancelled.")
        return
    
    migrator = ComprehensiveMongoMigrator(mongodb_url, database_name)
    
    try:
        # Connect to MongoDB
        await migrator.connect()
        
        # Run all migrations
        await migrator.migrate_category_hierarchies()
        await migrator.migrate_training_examples() 
        await migrator.migrate_accuracy_results()
        await migrator.migrate_system_configuration()
        await migrator.migrate_performance_reports()
        await migrator.create_indexes()
        
        # Show summary
        await migrator.print_migration_summary()
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
    
    finally:
        await migrator.close()

if __name__ == "__main__":
    if not PYMONGO_AVAILABLE:
        print("Please install MongoDB driver first:")
        print("pip install motor pymongo")
    else:
        asyncio.run(main())
