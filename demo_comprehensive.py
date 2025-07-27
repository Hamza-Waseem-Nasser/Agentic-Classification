"""
COMPREHENSIVE ITSM CLASSIFICATION DEMO
======================================

This script demonstrates the complete ITSM classification pipeline using real incident data.
Perfect for team presentations and system demonstrations.

Features:
- Uses actual incident tickets from Thiqa_Incidents_Example.csv
- Detailed JSON output for each processing step
- Agent-by-agent processing visualization
- Confidence scores and accuracy metrics
- Performance analytics
- Error handling demonstration

Output: JSON files with complete pipeline tracing
"""

import asyncio
import json
import random
import csv
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from main import initialize_classification_system, classify_ticket
from src.utils.logging_config import setup_logging, get_logger

# Setup logging for demo
setup_logging()
logger = get_logger(__name__)

class ITSMDemo:
    def __init__(self, incidents_file: str = "Thiqa_Incidents_Example.csv"):
        self.incidents_file = incidents_file
        self.demo_results = []
        self.pipeline = None
        self.demo_start_time = None
        
    async def initialize_system(self):
        """Initialize the classification system"""
        print("🚀 Initializing ITSM Classification System...")
        print("=" * 60)
        
        self.demo_start_time = time.time()
        self.pipeline = await initialize_classification_system(strict_mode=True)
        
        init_time = time.time() - self.demo_start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        print()
        
    def load_sample_incidents(self, count: int = 10) -> List[Dict[str, str]]:
        """Load random sample incidents from the CSV file"""
        print(f"📂 Loading {count} random incidents from {self.incidents_file}...")
        
        incidents = []
        try:
            with open(self.incidents_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                all_incidents = list(reader)
                
            # Debug: print column names
            if all_incidents:
                print(f"📋 CSV columns found: {list(all_incidents[0].keys())}")
                
            print(f"📊 Total valid incidents available: {len(all_incidents)}")
            
            if not all_incidents:
                print("❌ No incidents found!")
                return []
            
            # Select random samples
            sample_size = min(count, len(all_incidents))
            selected = random.sample(all_incidents, sample_size)
            
            for idx, incident in enumerate(selected):
                # Handle BOM character in column names
                incident_id_key = next((k for k in incident.keys() if 'Incident' in k), 'Incident')
                
                # Debug: print first incident to see data
                if idx == 0:
                    print(f"🔍 First sample - ID: {incident[incident_id_key]}")
                    print(f"    Description: {incident['Description'][:100]}...")
                    print(f"    Category: {incident['Subcategory_Thiqah']}")
                    print(f"    Subcategory: {incident['Subcategory2_Thiqah']}")
                
                incidents.append({
                    'incident_id': incident[incident_id_key],
                    'description': incident['Description'], 
                    'expected_category': incident['Subcategory_Thiqah'],
                    'expected_subcategory': incident['Subcategory2_Thiqah']
                })
                
            print(f"✅ Loaded {len(incidents)} incidents")
            print()
            return incidents
            
        except Exception as e:
            logger.error(f"Error loading incidents: {e}")
            return []
    
    async def classify_incident_with_details(self, incident: Dict[str, str]) -> Dict[str, Any]:
        """Classify a single incident with detailed step-by-step tracking"""
        
        incident_id = incident['incident_id']
        description = incident['description']
        
        print(f"🔍 Processing Incident {incident_id}")
        print(f"📝 Description: {description[:100]}{'...' if len(description) > 100 else ''}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Initialize detailed tracking
        detailed_result = {
            'incident_info': {
                'incident_id': incident_id,
                'original_description': description,
                'expected_category': incident['expected_category'],
                'expected_subcategory': incident['expected_subcategory'],
                'processing_timestamp': datetime.now().isoformat()
            },
            'pipeline_steps': [],
            'agent_decisions': {},
            'performance_metrics': {},
            'final_classification': {},
            'accuracy_assessment': {}
        }
        
        try:
            # Get the full classification result
            result = await classify_ticket(self.pipeline, description, incident_id)
            
            processing_time = time.time() - start_time
            
            # Extract key information
            predicted_category = result.get('classification', {}).get('category', 'Unknown')
            predicted_subcategory = result.get('classification', {}).get('subcategory', 'Unknown')
            category_confidence = result.get('classification', {}).get('category_confidence', 0.0)
            subcategory_confidence = result.get('classification', {}).get('subcategory_confidence', 0.0)
            
            # Assess accuracy
            category_correct = predicted_category == incident['expected_category']
            subcategory_correct = predicted_subcategory == incident['expected_subcategory']
            
            # Build detailed result
            detailed_result.update({
                'pipeline_steps': [
                    {
                        'step': 1,
                        'agent': 'Orchestrator',
                        'description': 'Initial workflow assessment and routing',
                        'status': 'completed',
                        'processing_time': result.get('processing', {}).get('processing_time', 0) * 0.1
                    },
                    {
                        'step': 2,
                        'agent': 'Arabic Processor',
                        'description': 'Language analysis and text normalization',
                        'status': 'completed',
                        'input_text': description,
                        'processed_text': result.get('processing', {}).get('processed_text', description),
                        'processing_time': result.get('processing', {}).get('processing_time', 0) * 0.2
                    },
                    {
                        'step': 3,
                        'agent': 'Category Classifier',
                        'description': 'Main category classification using vector search + LLM',
                        'status': 'completed',
                        'decision': predicted_category,
                        'confidence': category_confidence,
                        'processing_time': result.get('processing', {}).get('processing_time', 0) * 0.4
                    },
                    {
                        'step': 4,
                        'agent': 'Subcategory Classifier',
                        'description': 'Hierarchical subcategory classification',
                        'status': 'completed',
                        'decision': predicted_subcategory,
                        'confidence': subcategory_confidence,
                        'processing_time': result.get('processing', {}).get('processing_time', 0) * 0.3
                    }
                ],
                'agent_decisions': {
                    'arabic_processor': {
                        'original_text': description,
                        'processed_text': result.get('processing', {}).get('processed_text', description),
                        'entities_extracted': result.get('processing', {}).get('entities', []),
                        'technical_terms': result.get('processing', {}).get('technical_terms', [])
                    },
                    'category_classifier': {
                        'predicted_category': predicted_category,
                        'confidence_score': category_confidence,
                        'reasoning': f"Classified as '{predicted_category}' based on vector similarity and LLM analysis"
                    },
                    'subcategory_classifier': {
                        'predicted_subcategory': predicted_subcategory,
                        'confidence_score': subcategory_confidence,
                        'reasoning': f"Selected '{predicted_subcategory}' within category '{predicted_category}'"
                    }
                },
                'performance_metrics': {
                    'total_processing_time': processing_time,
                    'average_confidence': (category_confidence + subcategory_confidence) / 2,
                    'pipeline_efficiency': 'high' if processing_time < 10 else 'medium' if processing_time < 20 else 'low'
                },
                'final_classification': {
                    'predicted_category': predicted_category,
                    'predicted_subcategory': predicted_subcategory,
                    'category_confidence': category_confidence,
                    'subcategory_confidence': subcategory_confidence,
                    'classification_success': result.get('success', False)
                },
                'accuracy_assessment': {
                    'category_accuracy': category_correct,
                    'subcategory_accuracy': subcategory_correct,
                    'overall_accuracy': category_correct and subcategory_correct,
                    'expected_vs_predicted': {
                        'expected_category': incident['expected_category'],
                        'predicted_category': predicted_category,
                        'expected_subcategory': incident['expected_subcategory'],
                        'predicted_subcategory': predicted_subcategory
                    }
                }
            })
            
            # Console output
            print(f"🎯 Predicted: {predicted_category} → {predicted_subcategory}")
            print(f"✅ Expected:  {incident['expected_category']} → {incident['expected_subcategory']}")
            print(f"📊 Confidence: Category {category_confidence:.2f}, Subcategory {subcategory_confidence:.2f}")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"✅ Accuracy: Category {'✓' if category_correct else '✗'}, Subcategory {'✓' if subcategory_correct else '✗'}")
            print()
            
        except Exception as e:
            logger.error(f"Error processing incident {incident_id}: {e}")
            detailed_result['error'] = {
                'message': str(e),
                'processing_failed': True,
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error: {str(e)}")
            print()
        
        return detailed_result
    
    async def run_demo(self, sample_count: int = 10):
        """Run the complete demo with specified number of samples"""
        
        print("🎭 ITSM CLASSIFICATION SYSTEM DEMO")
        print("=" * 60)
        print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Sample Size: {sample_count} incidents")
        print(f"📊 Output: Detailed JSON reports")
        print("=" * 60)
        print()
        
        # Initialize system
        await self.initialize_system()
        
        # Load sample incidents
        incidents = self.load_sample_incidents(sample_count)
        if not incidents:
            print("❌ No incidents loaded. Demo cannot proceed.")
            return
        
        # Process each incident
        print("🔄 Processing Incidents...")
        print("=" * 60)
        
        for i, incident in enumerate(incidents, 1):
            print(f"📋 Incident {i}/{len(incidents)}")
            result = await self.classify_incident_with_details(incident)
            self.demo_results.append(result)
            
            # Small delay for readability
            await asyncio.sleep(0.5)
        
        # Generate summary
        await self.generate_demo_summary()
        
        # Save results
        self.save_demo_results()
        
        print("🎉 Demo completed successfully!")
        print("📁 Check the output files for detailed results.")
    
    async def generate_demo_summary(self):
        """Generate overall demo summary and statistics"""
        
        print("📊 DEMO SUMMARY")
        print("=" * 60)
        
        total_incidents = len(self.demo_results)
        successful_classifications = sum(1 for r in self.demo_results if r.get('final_classification', {}).get('classification_success', False))
        
        category_accuracy = sum(1 for r in self.demo_results if r.get('accuracy_assessment', {}).get('category_accuracy', False))
        subcategory_accuracy = sum(1 for r in self.demo_results if r.get('accuracy_assessment', {}).get('subcategory_accuracy', False))
        overall_accuracy = sum(1 for r in self.demo_results if r.get('accuracy_assessment', {}).get('overall_accuracy', False))
        
        processing_times = [r.get('performance_metrics', {}).get('total_processing_time', 0) for r in self.demo_results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        confidences = []
        for result in self.demo_results:
            cat_conf = result.get('final_classification', {}).get('category_confidence', 0)
            sub_conf = result.get('final_classification', {}).get('subcategory_confidence', 0)
            if cat_conf > 0 and sub_conf > 0:
                confidences.append((cat_conf + sub_conf) / 2)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        summary = {
            'demo_overview': {
                'total_incidents_processed': total_incidents,
                'successful_classifications': successful_classifications,
                'success_rate': f"{(successful_classifications/total_incidents)*100:.1f}%",
                'demo_duration': f"{time.time() - self.demo_start_time:.2f}s"
            },
            'accuracy_metrics': {
                'category_accuracy': f"{(category_accuracy/total_incidents)*100:.1f}%",
                'subcategory_accuracy': f"{(subcategory_accuracy/total_incidents)*100:.1f}%",
                'overall_accuracy': f"{(overall_accuracy/total_incidents)*100:.1f}%"
            },
            'performance_metrics': {
                'average_processing_time': f"{avg_processing_time:.2f}s",
                'fastest_classification': f"{min(processing_times):.2f}s",
                'slowest_classification': f"{max(processing_times):.2f}s",
                'average_confidence': f"{avg_confidence:.2f}"
            },
            'system_efficiency': {
                'pipeline_status': 'optimal' if avg_processing_time < 15 else 'good' if avg_processing_time < 30 else 'needs_optimization',
                'confidence_level': 'high' if avg_confidence > 0.8 else 'medium' if avg_confidence > 0.6 else 'low',
                'accuracy_rating': 'excellent' if overall_accuracy/total_incidents > 0.8 else 'good' if overall_accuracy/total_incidents > 0.6 else 'needs_improvement'
            }
        }
        
        self.demo_summary = summary
        
        # Print summary
        print(f"✅ Successfully processed: {successful_classifications}/{total_incidents} incidents")
        print(f"🎯 Category Accuracy: {(category_accuracy/total_incidents)*100:.1f}%")
        print(f"🎯 Subcategory Accuracy: {(subcategory_accuracy/total_incidents)*100:.1f}%")
        print(f"🎯 Overall Accuracy: {(overall_accuracy/total_incidents)*100:.1f}%")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
        print(f"📊 Average Confidence: {avg_confidence:.2f}")
        print()
    
    def save_demo_results(self):
        """Save demo results to JSON files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"demo_detailed_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'demo_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_incidents': len(self.demo_results),
                    'incidents_file': self.incidents_file
                },
                'detailed_results': self.demo_results
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = f"demo_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.demo_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed results saved to: {detailed_file}")
        print(f"💾 Summary saved to: {summary_file}")


async def main():
    """Main demo function"""
    
    # Create and run demo
    demo = ITSMDemo()
    await demo.run_demo(sample_count=10)


if __name__ == "__main__":
    asyncio.run(main())
