"""
Test the accuracy endpoint of your API
=====================================

This script tests the /test-accuracy endpoint that runs your accuracy tests.
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_accuracy_endpoint():
    print("🧪 Testing ITSM Classification Accuracy Endpoint")
    print("=" * 50)
    
    # Test with small number of cases first
    test_cases = [5, 10, 20]
    
    for max_cases in test_cases:
        print(f"\n📊 Testing with {max_cases} cases...")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/test-accuracy",
                params={"max_cases": max_cases},
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('test_summary', {})
                
                print(f"✅ Test completed in {end_time - start_time:.1f} seconds")
                print(f"📈 Total Cases: {summary.get('total_cases', 0)}")
                print(f"✅ Successful: {summary.get('successful', 0)}")
                print(f"❌ Failed: {summary.get('failed', 0)}")
                
                accuracy = summary.get('accuracy', {})
                print(f"🎯 Category Accuracy: {accuracy.get('category_accuracy', '0%')}")
                print(f"🎯 Subcategory Accuracy: {accuracy.get('subcategory_accuracy', '0%')}")
                print(f"🎯 Overall Accuracy: {accuracy.get('overall_accuracy', '0%')}")
                
                # Show sample results
                print(f"\n📋 Sample Results:")
                sample_results = result.get('sample_results', [])
                for i, sample in enumerate(sample_results[:3], 1):
                    if sample.get('success', False):
                        print(f"  {i}. {sample.get('incident_id', 'N/A')}: "
                              f"{sample.get('predicted_category', 'N/A')} → "
                              f"{sample.get('predicted_subcategory', 'N/A')} "
                              f"({'✓' if sample.get('overall_correct', False) else '✗'})")
                    else:
                        print(f"  {i}. {sample.get('incident_id', 'N/A')}: Error - {sample.get('error', 'Unknown')}")
                
            else:
                print(f"❌ Test failed with status {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        print("-" * 50)

def show_curl_example():
    print("\n🔧 CURL Example for your CTO:")
    print("=" * 40)
    print("# Run accuracy test with 10 cases")
    print(f"curl -X POST '{API_URL}/test-accuracy?max_cases=10' -H 'Content-Type: application/json'")
    print()
    print("# Run accuracy test with 50 cases")
    print(f"curl -X POST '{API_URL}/test-accuracy?max_cases=50' -H 'Content-Type: application/json'")

if __name__ == "__main__":
    print("🚀 ITSM Classification - Accuracy Test API")
    print("Make sure your simple_api.py is running on http://localhost:8000")
    print()
    
    choice = input("Press Enter to run accuracy tests, or 'c' for curl examples: ").strip().lower()
    
    if choice == 'c':
        show_curl_example()
    else:
        test_accuracy_endpoint()
        show_curl_example()
