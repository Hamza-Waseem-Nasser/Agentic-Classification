"""
ITSM CLASSIFICATION ACCURACY TESTING
===================================

This script tests your classification system against the "Correct User Descriptions.csv" file
and provides comprehensive accuracy metrics and improvement recommendations.

Features:
- Processes all test cases from the CSV file
- Calculates accuracy scores for categories and subcategories
- Identifies misclassified cases for improvement
- Provides detailed error analysis
- Suggests specific improvements for embeddings and prompts
- Generates comprehensive reports
"""

import asyncio
import csv
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

from main import initialize_classification_system, classify_ticket
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

class ClassificationAccuracyTester:
    """Test classification accuracy against ground truth data"""
    
    def __init__(self, test_file: str = "Correct User Descrioptions.csv"):
        self.test_file = test_file
        self.pipeline = None
        self.test_results = []
        self.accuracy_metrics = {}
        self.error_analysis = {}
        self.improvement_suggestions = []
        
    async def initialize_system(self):
        """Initialize the classification system"""
        print("🚀 Initializing ITSM Classification System for Testing...")
        print("=" * 60)
        
        start_time = time.time()
        self.pipeline = await initialize_classification_system(strict_mode=True)
        
        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        print()
        
    def load_test_data(self) -> List[Dict[str, str]]:
        """Load test data from CSV file"""
        print(f"📂 Loading test data from {self.test_file}...")
        
        test_cases = []
        try:
            # Read with pandas to handle encoding better
            df = pd.read_csv(self.test_file, encoding='utf-8')
            
            print(f"📋 Columns found: {list(df.columns)}")
            print(f"📊 Total test cases: {len(df)}")
            
            # Handle the typo in column name
            subcategory_col = 'Subcatgory' if 'Subcatgory' in df.columns else 'Subcategory'
            
            for idx, row in df.iterrows():
                test_cases.append({
                    'incident_id': str(row['Incident']),
                    'description': str(row['Description']),
                    'expected_category': str(row['Category']),
                    'expected_subcategory': str(row[subcategory_col]) if pd.notna(row[subcategory_col]) else ''
                })
            
            print(f"✅ Loaded {len(test_cases)} test cases")
            print(f"🔍 Sample case - ID: {test_cases[0]['incident_id']}")
            print(f"    Description: {test_cases[0]['description'][:100]}...")
            print(f"    Expected: {test_cases[0]['expected_category']} → {test_cases[0]['expected_subcategory']}")
            print()
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return []
    
    async def test_single_case(self, test_case: Dict[str, str]) -> Dict[str, Any]:
        """Test a single classification case"""
        
        incident_id = test_case['incident_id']
        description = test_case['description']
        expected_category = test_case['expected_category']
        expected_subcategory = test_case['expected_subcategory']
        
        start_time = time.time()
        
        try:
            # Classify the ticket
            result = await classify_ticket(self.pipeline, description, incident_id)
            
            processing_time = time.time() - start_time
            
            # Extract predictions
            predicted_category = result.get('classification', {}).get('category', 'Unknown')
            predicted_subcategory = result.get('classification', {}).get('subcategory', 'Unknown')
            category_confidence = result.get('classification', {}).get('category_confidence', 0.0)
            subcategory_confidence = result.get('classification', {}).get('subcategory_confidence', 0.0)
            
            # Assess accuracy
            category_correct = predicted_category == expected_category
            subcategory_correct = predicted_subcategory == expected_subcategory
            overall_correct = category_correct and subcategory_correct
            
            # Build test result
            test_result = {
                'incident_id': incident_id,
                'description': description,
                'expected_category': expected_category,
                'expected_subcategory': expected_subcategory,
                'predicted_category': predicted_category,
                'predicted_subcategory': predicted_subcategory,
                'category_confidence': category_confidence,
                'subcategory_confidence': subcategory_confidence,
                'category_correct': category_correct,
                'subcategory_correct': subcategory_correct,
                'overall_correct': overall_correct,
                'processing_time': processing_time,
                'success': result.get('success', False),
                'classification_details': result.get('classification', {}),
                'processing_details': result.get('processing', {})
            }
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing case {incident_id}: {e}")
            return {
                'incident_id': incident_id,
                'description': description,
                'expected_category': expected_category,
                'expected_subcategory': expected_subcategory,
                'error': str(e),
                'success': False
            }
    
    async def run_accuracy_test(self, max_cases: int = None):
        """Run accuracy test on all or subset of test cases"""
        
        print("🧪 ITSM CLASSIFICATION ACCURACY TEST")
        print("=" * 60)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Test File: {self.test_file}")
        print("=" * 60)
        print()
        
        # Initialize system
        await self.initialize_system()
        
        # Load test data
        test_cases = self.load_test_data()
        if not test_cases:
            print("❌ No test cases loaded. Test cannot proceed.")
            return
        
        # Limit test cases if specified
        if max_cases and max_cases < len(test_cases):
            test_cases = test_cases[:max_cases]
            print(f"🔢 Testing limited to {max_cases} cases")
            print()
        
        # Run tests
        print(f"🔄 Testing {len(test_cases)} cases...")
        print("=" * 60)
        
        total_cases = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            if i % 50 == 0 or i == 1:  # Progress updates
                print(f"📋 Progress: {i}/{total_cases} ({(i/total_cases)*100:.1f}%)")
            
            result = await self.test_single_case(test_case)
            self.test_results.append(result)
            
            # Small delay to avoid overwhelming the system
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        print(f"✅ Testing completed: {len(self.test_results)} cases processed")
        print()
        
        # Calculate metrics
        self.calculate_accuracy_metrics()
        
        # Analyze errors
        self.analyze_classification_errors()
        
        # Generate improvement suggestions
        self.generate_improvement_suggestions()
        
        # Save results
        self.save_test_results()
        
        # Print summary
        self.print_test_summary()
    
    def calculate_accuracy_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        
        successful_results = [r for r in self.test_results if r.get('success', False)]
        total_cases = len(successful_results)
        
        if total_cases == 0:
            print("❌ No successful classifications to analyze")
            return
        
        # Basic accuracy metrics
        category_correct = sum(1 for r in successful_results if r.get('category_correct', False))
        subcategory_correct = sum(1 for r in successful_results if r.get('subcategory_correct', False))
        overall_correct = sum(1 for r in successful_results if r.get('overall_correct', False))
        
        # Confidence metrics
        confidences = []
        category_confidences = []
        subcategory_confidences = []
        
        for result in successful_results:
            cat_conf = result.get('category_confidence', 0)
            sub_conf = result.get('subcategory_confidence', 0)
            
            category_confidences.append(cat_conf)
            subcategory_confidences.append(sub_conf)
            
            if cat_conf > 0 and sub_conf > 0:
                confidences.append((cat_conf + sub_conf) / 2)
        
        # Performance metrics
        processing_times = [r.get('processing_time', 0) for r in successful_results if r.get('processing_time')]
        
        self.accuracy_metrics = {
            'basic_metrics': {
                'total_test_cases': len(self.test_results),
                'successful_classifications': total_cases,
                'failed_classifications': len(self.test_results) - total_cases,
                'success_rate': f"{(total_cases/len(self.test_results))*100:.1f}%"
            },
            'accuracy_scores': {
                'category_accuracy': f"{(category_correct/total_cases)*100:.1f}%",
                'subcategory_accuracy': f"{(subcategory_correct/total_cases)*100:.1f}%",
                'overall_accuracy': f"{(overall_correct/total_cases)*100:.1f}%",
                'category_correct_count': f"{category_correct}/{total_cases}",
                'subcategory_correct_count': f"{subcategory_correct}/{total_cases}",
                'overall_correct_count': f"{overall_correct}/{total_cases}"
            },
            'confidence_metrics': {
                'average_category_confidence': f"{sum(category_confidences)/len(category_confidences):.3f}" if category_confidences else "0.000",
                'average_subcategory_confidence': f"{sum(subcategory_confidences)/len(subcategory_confidences):.3f}" if subcategory_confidences else "0.000",
                'average_overall_confidence': f"{sum(confidences)/len(confidences):.3f}" if confidences else "0.000",
                'high_confidence_cases': sum(1 for c in confidences if c > 0.8),
                'medium_confidence_cases': sum(1 for c in confidences if 0.6 <= c <= 0.8),
                'low_confidence_cases': sum(1 for c in confidences if c < 0.6)
            },
            'performance_metrics': {
                'average_processing_time': f"{sum(processing_times)/len(processing_times):.2f}s" if processing_times else "0.00s",
                'fastest_classification': f"{min(processing_times):.2f}s" if processing_times else "0.00s",
                'slowest_classification': f"{max(processing_times):.2f}s" if processing_times else "0.00s",
                'total_test_time': f"{sum(processing_times):.2f}s" if processing_times else "0.00s"
            }
        }
    
    def analyze_classification_errors(self):
        """Analyze classification errors to identify patterns"""
        
        successful_results = [r for r in self.test_results if r.get('success', False)]
        
        # Category errors
        category_errors = [r for r in successful_results if not r.get('category_correct', False)]
        subcategory_errors = [r for r in successful_results if not r.get('subcategory_correct', False)]
        
        # Most confused categories
        category_confusion = defaultdict(Counter)
        subcategory_confusion = defaultdict(Counter)
        
        for error in category_errors:
            expected = error['expected_category']
            predicted = error['predicted_category']
            category_confusion[expected][predicted] += 1
        
        for error in subcategory_errors:
            expected = error['expected_subcategory']
            predicted = error['predicted_subcategory']
            subcategory_confusion[expected][predicted] += 1
        
        # Low confidence correct classifications (potential issues)
        low_confidence_correct = [
            r for r in successful_results 
            if r.get('overall_correct', False) and 
            (r.get('category_confidence', 0) + r.get('subcategory_confidence', 0)) / 2 < 0.7
        ]
        
        # High confidence incorrect classifications (concerning)
        high_confidence_incorrect = [
            r for r in successful_results 
            if not r.get('overall_correct', False) and 
            (r.get('category_confidence', 0) + r.get('subcategory_confidence', 0)) / 2 > 0.8
        ]
        
        self.error_analysis = {
            'category_errors': {
                'total_errors': len(category_errors),
                'error_rate': f"{(len(category_errors)/len(successful_results))*100:.1f}%",
                'most_confused_categories': dict(category_confusion),
                'sample_errors': category_errors[:5]  # First 5 errors as examples
            },
            'subcategory_errors': {
                'total_errors': len(subcategory_errors),
                'error_rate': f"{(len(subcategory_errors)/len(successful_results))*100:.1f}%",
                'most_confused_subcategories': dict(subcategory_confusion),
                'sample_errors': subcategory_errors[:5]
            },
            'confidence_issues': {
                'low_confidence_correct': {
                    'count': len(low_confidence_correct),
                    'description': "Cases correctly classified but with low confidence",
                    'samples': low_confidence_correct[:3]
                },
                'high_confidence_incorrect': {
                    'count': len(high_confidence_incorrect),
                    'description': "Cases incorrectly classified but with high confidence",
                    'samples': high_confidence_incorrect[:3]
                }
            }
        }
    
    def generate_improvement_suggestions(self):
        """Generate specific suggestions for improving the classification system"""
        
        suggestions = []
        
        # Analyze accuracy metrics
        category_accuracy = float(self.accuracy_metrics['accuracy_scores']['category_accuracy'].rstrip('%'))
        subcategory_accuracy = float(self.accuracy_metrics['accuracy_scores']['subcategory_accuracy'].rstrip('%'))
        avg_confidence = float(self.accuracy_metrics['confidence_metrics']['average_overall_confidence'])
        
        # Category accuracy suggestions
        if category_accuracy < 80:
            suggestions.append({
                'area': 'Category Classification',
                'priority': 'High',
                'issue': f'Category accuracy is {category_accuracy:.1f}%, below 80% threshold',
                'suggestions': [
                    'Add more diverse training examples for category embeddings',
                    'Enhance category descriptions with more detailed keywords',
                    'Improve chain-of-thought prompts for category classification',
                    'Consider adjusting similarity thresholds in vector search'
                ]
            })
        
        # Subcategory accuracy suggestions  
        if subcategory_accuracy < 75:
            suggestions.append({
                'area': 'Subcategory Classification',
                'priority': 'High',
                'issue': f'Subcategory accuracy is {subcategory_accuracy:.1f}%, below 75% threshold',
                'suggestions': [
                    'Add more subcategory-specific training examples',
                    'Enhance hierarchical context in subcategory prompts',
                    'Review subcategory descriptions for clarity',
                    'Consider parent category context in subcategory classification'
                ]
            })
        
        # Confidence-based suggestions
        if avg_confidence < 0.7:
            suggestions.append({
                'area': 'Confidence Scores',
                'priority': 'Medium',
                'issue': f'Average confidence is {avg_confidence:.3f}, below 0.7 threshold',
                'suggestions': [
                    'Improve embedding quality with more representative examples',
                    'Enhance LLM prompts with clearer instructions',
                    'Add confidence calibration based on historical performance',
                    'Consider ensemble methods for more reliable predictions'
                ]
            })
        
        # Error pattern analysis
        category_errors = self.error_analysis.get('category_errors', {})
        if category_errors.get('total_errors', 0) > 0:
            confused_categories = category_errors.get('most_confused_categories', {})
            for expected_cat, predictions in confused_categories.items():
                most_confused_with = predictions.most_common(1)[0] if predictions else None
                if most_confused_with:
                    predicted_cat, error_count = most_confused_with
                    suggestions.append({
                        'area': 'Specific Category Confusion',
                        'priority': 'Medium',
                        'issue': f'"{expected_cat}" often confused with "{predicted_cat}" ({error_count} times)',
                        'suggestions': [
                            f'Add distinguishing examples between "{expected_cat}" and "{predicted_cat}"',
                            f'Enhance description clarity for "{expected_cat}" category',
                            f'Review and improve keywords for "{expected_cat}"',
                            'Add negative examples in prompts to avoid confusion'
                        ]
                    })
        
        # High confidence errors (most concerning)
        high_conf_errors = self.error_analysis.get('confidence_issues', {}).get('high_confidence_incorrect', {})
        if high_conf_errors.get('count', 0) > 0:
            suggestions.append({
                'area': 'High Confidence Errors',
                'priority': 'Critical',
                'issue': f'{high_conf_errors["count"]} cases incorrectly classified with high confidence',
                'suggestions': [
                    'Review these specific cases for systematic bias in training data',
                    'Add these misclassified examples as negative examples',
                    'Improve prompt engineering to reduce overconfidence',
                    'Consider additional validation steps for high-confidence predictions'
                ]
            })
        
        self.improvement_suggestions = suggestions
    
    def save_test_results(self):
        """Save comprehensive test results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"accuracy_test_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'test_file': self.test_file,
                    'total_cases': len(self.test_results)
                },
                'accuracy_metrics': self.accuracy_metrics,
                'error_analysis': self.error_analysis,
                'improvement_suggestions': self.improvement_suggestions,
                'detailed_results': self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        summary_file = f"accuracy_test_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ITSM CLASSIFICATION ACCURACY TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test File: {self.test_file}\n")
            f.write(f"Total Cases: {len(self.test_results)}\n\n")
            
            # Accuracy metrics
            f.write("ACCURACY METRICS:\n")
            f.write("-" * 20 + "\n")
            for category, metrics in self.accuracy_metrics.items():
                f.write(f"\n{category.upper()}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
            
            # Improvement suggestions
            f.write("\n\nIMPROVEMENT SUGGESTIONS:\n")
            f.write("-" * 25 + "\n")
            for i, suggestion in enumerate(self.improvement_suggestions, 1):
                f.write(f"\n{i}. {suggestion['area']} (Priority: {suggestion['priority']})\n")
                f.write(f"   Issue: {suggestion['issue']}\n")
                f.write("   Suggestions:\n")
                for sug in suggestion['suggestions']:
                    f.write(f"   - {sug}\n")
        
        print(f"💾 Detailed results saved to: {detailed_file}")
        print(f"💾 Summary saved to: {summary_file}")
    
    def print_test_summary(self):
        """Print test summary to console"""
        
        print("\n" + "🏆 TEST SUMMARY" + "\n")
        print("=" * 60)
        
        # Basic metrics
        basic = self.accuracy_metrics.get('basic_metrics', {})
        accuracy = self.accuracy_metrics.get('accuracy_scores', {})
        confidence = self.accuracy_metrics.get('confidence_metrics', {})
        
        print(f"📊 Total Test Cases: {basic.get('total_test_cases', 0)}")
        print(f"✅ Successful Classifications: {basic.get('successful_classifications', 0)}")
        print(f"❌ Failed Classifications: {basic.get('failed_classifications', 0)}")
        print(f"📈 Success Rate: {basic.get('success_rate', '0%')}")
        print()
        
        print("🎯 ACCURACY SCORES:")
        print(f"   Category Accuracy: {accuracy.get('category_accuracy', '0%')}")
        print(f"   Subcategory Accuracy: {accuracy.get('subcategory_accuracy', '0%')}")
        print(f"   Overall Accuracy: {accuracy.get('overall_accuracy', '0%')}")
        print()
        
        print("🔍 CONFIDENCE METRICS:")
        print(f"   Average Category Confidence: {confidence.get('average_category_confidence', '0.000')}")
        print(f"   Average Subcategory Confidence: {confidence.get('average_subcategory_confidence', '0.000')}")
        print(f"   Average Overall Confidence: {confidence.get('average_overall_confidence', '0.000')}")
        print()
        
        # Top improvement suggestions
        print("💡 TOP IMPROVEMENT SUGGESTIONS:")
        for i, suggestion in enumerate(self.improvement_suggestions[:3], 1):
            print(f"   {i}. {suggestion['area']} (Priority: {suggestion['priority']})")
            print(f"      {suggestion['issue']}")
        
        print("\n📁 Check the saved files for detailed analysis and complete suggestions.")
        print("=" * 60)


async def main():
    """Main function for accuracy testing"""
    
    # Get user input for test parameters
    print("🧪 ITSM Classification Accuracy Tester")
    print("=" * 40)
    print()
    
    max_cases_input = input("Enter max test cases (or press Enter for all 616 cases): ").strip()
    max_cases = int(max_cases_input) if max_cases_input.isdigit() else None
    
    if max_cases:
        print(f"🔢 Testing limited to {max_cases} cases")
    else:
        print("🔢 Testing all available cases (~616)")
    
    print()
    response = input("Continue with accuracy test? (y/n): ").lower().strip()
    if response != 'y':
        print("Test cancelled.")
        return
    
    # Run the test
    tester = ClassificationAccuracyTester()
    await tester.run_accuracy_test(max_cases=max_cases)


if __name__ == "__main__":
    asyncio.run(main())
