"""
CATEGORY VALIDATION SCRIPT
=========================

Quick script to validate that test file categories match your system's categories.
"""

import asyncio
import pandas as pd
from main import initialize_classification_system

async def validate_categories():
    """Validate test categories against system categories"""
    
    print("🔍 Validating test categories against system categories...")
    print("=" * 60)
    
    # Load test data
    df = pd.read_csv('Correct User Descrioptions.csv')
    test_categories = set(df['Category'].unique())
    test_subcategories = set(df['Subcatgory'].dropna().unique())
    
    # Initialize system
    pipeline = await initialize_classification_system()
    system_categories = set(pipeline.hierarchy.categories.keys())
    
    system_subcategories = set()
    for category in pipeline.hierarchy.categories.values():
        system_subcategories.update(category.subcategories.keys())
    
    # Compare categories
    print("📊 CATEGORY ANALYSIS:")
    print(f"   Test file categories: {len(test_categories)}")
    print(f"   System categories: {len(system_categories)}")
    
    missing_in_system = test_categories - system_categories
    missing_in_test = system_categories - test_categories
    
    if missing_in_system:
        print(f"\n❌ Categories in test but NOT in system: {len(missing_in_system)}")
        for cat in sorted(missing_in_system):
            print(f"   - {cat}")
    
    if missing_in_test:
        print(f"\n⚠️  Categories in system but NOT in test: {len(missing_in_test)}")
        for cat in sorted(missing_in_test):
            print(f"   - {cat}")
    
    # Compare subcategories
    print(f"\n📊 SUBCATEGORY ANALYSIS:")
    print(f"   Test file subcategories: {len(test_subcategories)}")
    print(f"   System subcategories: {len(system_subcategories)}")
    
    missing_sub_in_system = test_subcategories - system_subcategories
    missing_sub_in_test = system_subcategories - test_subcategories
    
    if missing_sub_in_system:
        print(f"\n❌ Subcategories in test but NOT in system: {len(missing_sub_in_system)}")
        for sub in sorted(missing_sub_in_system):
            print(f"   - {sub}")
    
    if missing_sub_in_test:
        print(f"\n⚠️  Subcategories in system but NOT in test: {len(missing_sub_in_test)}")
        for sub in sorted(missing_sub_in_test):
            print(f"   - {sub}")
    
    # Overall validation
    print(f"\n✅ VALIDATION SUMMARY:")
    print(f"   Category match rate: {len(test_categories & system_categories)}/{len(test_categories)} ({(len(test_categories & system_categories)/len(test_categories))*100:.1f}%)")
    print(f"   Subcategory match rate: {len(test_subcategories & system_subcategories)}/{len(test_subcategories)} ({(len(test_subcategories & system_subcategories)/len(test_subcategories))*100:.1f}%)")
    
    if not missing_in_system and not missing_sub_in_system:
        print("\n🎉 Perfect match! All test categories and subcategories exist in your system.")
    elif len(missing_in_system) + len(missing_sub_in_system) <= 3:
        print("\n✅ Good match! Only minor discrepancies found.")
    else:
        print("\n⚠️  Significant discrepancies found. Consider updating your hierarchy or test data.")

if __name__ == "__main__":
    asyncio.run(validate_categories())
