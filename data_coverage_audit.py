"""
COMPREHENSIVE DATA COVERAGE AUDIT
=================================

This script analyzes the coverage between category definitions and training data
to provide precise statistics and identify gaps.
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json

class DataCoverageAuditor:
    def __init__(self):
        self.categories_df = None
        self.training_df = None
        self.coverage_report = {}
        
    def load_data(self):
        """Load both CSV files"""
        print("📂 Loading data files...")
        
        # Load category definitions
        self.categories_df = pd.read_csv("Category + SubCategory.csv", encoding='utf-8')
        print(f"✅ Category definitions loaded: {len(self.categories_df)} rows")
        
        # Load training data
        self.training_df = pd.read_csv("Correct User Descrioptions.csv", encoding='utf-8')
        print(f"✅ Training data loaded: {len(self.training_df)} rows")
        
        # Clean up column names
        if 'Subcatgory' in self.training_df.columns:
            self.training_df['Subcategory'] = self.training_df['Subcatgory']
            
        print("\n" + "="*60)
        
    def analyze_category_definitions(self):
        """Analyze the category definition structure"""
        print("🏗️ ANALYZING CATEGORY DEFINITIONS")
        print("="*60)
        
        # Get unique categories and subcategories
        defined_categories = self.categories_df['Category'].unique()
        defined_categories = [cat for cat in defined_categories if pd.notna(cat) and cat.strip()]
        
        # Count subcategories per category
        category_subcounts = {}
        total_subcategories = 0
        
        for category in defined_categories:
            category_data = self.categories_df[self.categories_df['Category'] == category]
            subcategories = category_data['SubCategory'].dropna()
            subcategories = [sub for sub in subcategories if sub.strip()]
            category_subcounts[category] = len(subcategories)
            total_subcategories += len(subcategories)
        
        print(f"📊 Total Categories Defined: {len(defined_categories)}")
        print(f"📊 Total Subcategories Defined: {total_subcategories}")
        print(f"📊 Average Subcategories per Category: {total_subcategories/len(defined_categories):.1f}")
        
        print("\n📋 Categories with Subcategory Counts:")
        for i, (category, count) in enumerate(sorted(category_subcounts.items(), key=lambda x: x[1], reverse=True), 1):
            print(f"  {i:2d}. {category}: {count} subcategories")
            
        self.coverage_report['defined_categories'] = defined_categories
        self.coverage_report['defined_subcategory_counts'] = category_subcounts
        self.coverage_report['total_defined_categories'] = len(defined_categories)
        self.coverage_report['total_defined_subcategories'] = total_subcategories
        
        return defined_categories, category_subcounts
    
    def analyze_training_data(self):
        """Analyze the training data coverage"""
        print("\n🎯 ANALYZING TRAINING DATA COVERAGE")
        print("="*60)
        
        # Get training categories and subcategories
        training_categories = self.training_df['Category'].dropna().str.strip()
        training_subcategories = self.training_df['Subcategory'].dropna().str.strip()
        
        # Count occurrences
        category_counts = Counter(training_categories)
        subcategory_counts = Counter(training_subcategories)
        
        # Category-Subcategory combinations
        combo_counts = Counter()
        for _, row in self.training_df.iterrows():
            if pd.notna(row['Category']) and pd.notna(row['Subcategory']):
                combo = f"{row['Category'].strip()} → {row['Subcategory'].strip()}"
                combo_counts[combo] += 1
        
        print(f"📊 Total Training Examples: {len(self.training_df)}")
        print(f"📊 Unique Categories in Training: {len(category_counts)}")
        print(f"📊 Unique Subcategories in Training: {len(subcategory_counts)}")
        print(f"📊 Unique Category-Subcategory Combinations: {len(combo_counts)}")
        
        print(f"\n📈 Training Data Distribution by Category:")
        for i, (category, count) in enumerate(category_counts.most_common(), 1):
            percentage = (count / len(self.training_df)) * 100
            print(f"  {i:2d}. {category}: {count} examples ({percentage:.1f}%)")
        
        print(f"\n📈 Most Common Subcategories:")
        for i, (subcategory, count) in enumerate(subcategory_counts.most_common(10), 1):
            percentage = (count / len(self.training_df)) * 100
            print(f"  {i:2d}. {subcategory}: {count} examples ({percentage:.1f}%)")
            
        self.coverage_report['training_category_counts'] = dict(category_counts)
        self.coverage_report['training_subcategory_counts'] = dict(subcategory_counts)
        self.coverage_report['training_combo_counts'] = dict(combo_counts)
        
        return category_counts, subcategory_counts, combo_counts
    
    def analyze_coverage_gaps(self, defined_categories, category_subcounts, training_category_counts):
        """Identify coverage gaps between definitions and training data"""
        print("\n🔍 COVERAGE GAP ANALYSIS")
        print("="*60)
        
        training_categories_set = set(training_category_counts.keys())
        defined_categories_set = set(defined_categories)
        
        # Find gaps
        missing_from_training = defined_categories_set - training_categories_set
        extra_in_training = training_categories_set - defined_categories_set
        covered_categories = defined_categories_set & training_categories_set
        
        print(f"✅ Categories Covered: {len(covered_categories)}/{len(defined_categories)} ({len(covered_categories)/len(defined_categories)*100:.1f}%)")
        print(f"❌ Categories Missing from Training: {len(missing_from_training)}")
        print(f"⚠️  Extra Categories in Training: {len(extra_in_training)}")
        
        if missing_from_training:
            print(f"\n❌ MISSING CATEGORIES (No Training Examples):")
            for i, category in enumerate(sorted(missing_from_training), 1):
                subcategory_count = category_subcounts.get(category, 0)
                print(f"  {i:2d}. {category} ({subcategory_count} subcategories defined)")
        
        if extra_in_training:
            print(f"\n⚠️  EXTRA CATEGORIES (Not in Definitions):")
            for i, category in enumerate(sorted(extra_in_training), 1):
                count = training_category_counts[category]
                print(f"  {i:2d}. {category} ({count} examples)")
        
        # Coverage quality analysis
        print(f"\n📊 COVERAGE QUALITY BY CATEGORY:")
        for category in sorted(covered_categories):
            training_count = training_category_counts[category]
            defined_subcategories = category_subcounts.get(category, 0)
            
            # Get actual subcategories used in training for this category
            category_training_data = self.training_df[self.training_df['Category'].str.strip() == category]
            actual_subcategories_used = len(category_training_data['Subcategory'].dropna().str.strip().unique())
            
            coverage_ratio = actual_subcategories_used / defined_subcategories if defined_subcategories > 0 else 0
            
            print(f"  • {category}:")
            print(f"    - Training Examples: {training_count}")
            print(f"    - Defined Subcategories: {defined_subcategories}")
            print(f"    - Used Subcategories: {actual_subcategories_used}")
            print(f"    - Subcategory Coverage: {coverage_ratio:.1%}")
            
        self.coverage_report['coverage_summary'] = {
            'total_defined': len(defined_categories),
            'covered_count': len(covered_categories),
            'missing_count': len(missing_from_training),
            'extra_count': len(extra_in_training),
            'coverage_percentage': len(covered_categories)/len(defined_categories)*100,
            'missing_categories': list(missing_from_training),
            'extra_categories': list(extra_in_training),
            'covered_categories': list(covered_categories)
        }
        
        return missing_from_training, covered_categories
    
    def analyze_data_imbalance(self, training_category_counts):
        """Analyze data imbalance issues"""
        print("\n⚖️  DATA IMBALANCE ANALYSIS")
        print("="*60)
        
        counts = list(training_category_counts.values())
        total_examples = sum(counts)
        
        # Statistical analysis
        mean_examples = np.mean(counts)
        median_examples = np.median(counts)
        std_examples = np.std(counts)
        min_examples = min(counts)
        max_examples = max(counts)
        
        print(f"📈 Distribution Statistics:")
        print(f"  • Total Examples: {total_examples}")
        print(f"  • Mean per Category: {mean_examples:.1f}")
        print(f"  • Median per Category: {median_examples:.1f}")
        print(f"  • Standard Deviation: {std_examples:.1f}")
        print(f"  • Min Examples: {min_examples}")
        print(f"  • Max Examples: {max_examples}")
        print(f"  • Imbalance Ratio (Max/Min): {max_examples/min_examples:.1f}x")
        
        # Identify severely underrepresented categories
        threshold_low = mean_examples * 0.3  # 30% below average
        threshold_high = mean_examples * 2.0  # 200% above average
        
        underrepresented = [(cat, count) for cat, count in training_category_counts.items() if count < threshold_low]
        overrepresented = [(cat, count) for cat, count in training_category_counts.items() if count > threshold_high]
        
        print(f"\n⚠️  Severely Underrepresented (< {threshold_low:.1f} examples):")
        for category, count in sorted(underrepresented, key=lambda x: x[1]):
            percentage = (count / total_examples) * 100
            print(f"  • {category}: {count} examples ({percentage:.1f}%)")
            
        print(f"\n📈 Overrepresented (> {threshold_high:.1f} examples):")
        for category, count in sorted(overrepresented, key=lambda x: x[1], reverse=True):
            percentage = (count / total_examples) * 100
            print(f"  • {category}: {count} examples ({percentage:.1f}%)")
        
        # Pareto analysis (80/20 rule)
        sorted_counts = sorted(training_category_counts.items(), key=lambda x: x[1], reverse=True)
        cumulative_percentage = 0
        pareto_categories = []
        
        for category, count in sorted_counts:
            cumulative_percentage += (count / total_examples) * 100
            pareto_categories.append((category, count, cumulative_percentage))
            if cumulative_percentage >= 80:
                break
        
        print(f"\n📊 Pareto Analysis (Categories covering 80% of examples):")
        for i, (category, count, cum_pct) in enumerate(pareto_categories, 1):
            pct = (count / total_examples) * 100
            print(f"  {i}. {category}: {count} examples ({pct:.1f}%, cumulative: {cum_pct:.1f}%)")
            
        self.coverage_report['imbalance_analysis'] = {
            'statistics': {
                'mean': mean_examples,
                'median': median_examples,
                'std': std_examples,
                'min': min_examples,
                'max': max_examples,
                'imbalance_ratio': max_examples/min_examples
            },
            'underrepresented': underrepresented,
            'overrepresented': overrepresented,
            'pareto_80_percent': pareto_categories
        }
    
    def generate_subcategory_analysis(self):
        """Detailed subcategory coverage analysis"""
        print("\n📋 DETAILED SUBCATEGORY ANALYSIS")
        print("="*60)
        
        # Get all defined subcategories with their categories
        defined_subcategories = {}
        for _, row in self.categories_df.iterrows():
            if pd.notna(row['Category']) and pd.notna(row['SubCategory']):
                category = row['Category'].strip()
                subcategory = row['SubCategory'].strip()
                if category not in defined_subcategories:
                    defined_subcategories[category] = set()
                defined_subcategories[category].add(subcategory)
        
        # Get all training subcategories with their categories
        training_subcategories = {}
        for _, row in self.training_df.iterrows():
            if pd.notna(row['Category']) and pd.notna(row['Subcategory']):
                category = row['Category'].strip()
                subcategory = row['Subcategory'].strip()
                if category not in training_subcategories:
                    training_subcategories[category] = set()
                training_subcategories[category].add(subcategory)
        
        # Analysis by category
        total_defined_subs = sum(len(subs) for subs in defined_subcategories.values())
        total_training_subs = sum(len(subs) for subs in training_subcategories.values())
        
        print(f"📊 Total Defined Subcategories: {total_defined_subs}")
        print(f"📊 Total Training Subcategories: {total_training_subs}")
        
        subcategory_coverage = {}
        for category in defined_subcategories:
            defined_subs = defined_subcategories[category]
            training_subs = training_subcategories.get(category, set())
            
            covered = defined_subs & training_subs
            missing = defined_subs - training_subs
            extra = training_subs - defined_subs
            
            coverage_pct = len(covered) / len(defined_subs) * 100 if defined_subs else 0
            
            subcategory_coverage[category] = {
                'defined_count': len(defined_subs),
                'training_count': len(training_subs),
                'covered_count': len(covered),
                'missing_count': len(missing),
                'extra_count': len(extra),
                'coverage_percentage': coverage_pct,
                'covered_subcategories': list(covered),
                'missing_subcategories': list(missing),
                'extra_subcategories': list(extra)
            }
            
            if category in training_subcategories:  # Only show categories with training data
                print(f"\n📂 {category}:")
                print(f"  • Defined Subcategories: {len(defined_subs)}")
                print(f"  • Training Subcategories: {len(training_subs)}")
                print(f"  • Coverage: {len(covered)}/{len(defined_subs)} ({coverage_pct:.1f}%)")
                
                if missing:
                    print(f"  • Missing from Training: {len(missing)}")
                    for sub in sorted(missing)[:3]:  # Show first 3
                        print(f"    - {sub}")
                    if len(missing) > 3:
                        print(f"    ... and {len(missing)-3} more")
        
        self.coverage_report['subcategory_analysis'] = subcategory_coverage
        
        # Overall subcategory coverage
        total_covered = sum(data['covered_count'] for data in subcategory_coverage.values())
        overall_subcategory_coverage = total_covered / total_defined_subs * 100
        
        print(f"\n📊 OVERALL SUBCATEGORY COVERAGE: {total_covered}/{total_defined_subs} ({overall_subcategory_coverage:.1f}%)")
        
        return subcategory_coverage
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        print("\n💡 RECOMMENDATIONS & ACTION ITEMS")
        print("="*60)
        
        coverage_pct = self.coverage_report['coverage_summary']['coverage_percentage']
        missing_count = self.coverage_report['coverage_summary']['missing_count']
        
        print("🎯 PRIORITY ACTIONS:")
        
        if coverage_pct < 70:
            print("🔴 HIGH PRIORITY - Critical Coverage Gap:")
            print(f"   • Only {coverage_pct:.1f}% category coverage")
            print(f"   • {missing_count} categories have ZERO training examples")
            print("   • Classification accuracy will be severely impacted")
            print("   • Recommend adding 5-10 examples per missing category")
        
        # Imbalance issues
        imbalance_ratio = self.coverage_report['imbalance_analysis']['statistics']['imbalance_ratio']
        if imbalance_ratio > 10:
            print(f"\n🟡 MEDIUM PRIORITY - Severe Data Imbalance:")
            print(f"   • Imbalance ratio: {imbalance_ratio:.1f}x")
            print("   • Top categories dominate training data")
            print("   • Consider data augmentation or weighted sampling")
        
        # Subcategory coverage
        subcategory_coverage = self.coverage_report.get('subcategory_analysis', {})
        poor_subcategory_coverage = [
            cat for cat, data in subcategory_coverage.items() 
            if data['coverage_percentage'] < 50 and cat in self.coverage_report['coverage_summary']['covered_categories']
        ]
        
        if poor_subcategory_coverage:
            print(f"\n🟡 MEDIUM PRIORITY - Poor Subcategory Coverage:")
            print("   • Categories with <50% subcategory coverage:")
            for cat in poor_subcategory_coverage[:5]:
                pct = subcategory_coverage[cat]['coverage_percentage']
                print(f"     - {cat}: {pct:.1f}%")
        
        print(f"\n✅ RECOMMENDED NEXT STEPS:")
        print("1. 🎯 Add training examples for missing categories")
        print("2. 📊 Balance data distribution with augmentation")
        print("3. 🔍 Collect real examples for underrepresented subcategories")
        print("4. ⚖️  Implement class weighting in your model")
        print("5. 📈 Monitor classification performance by category")
        
    def save_detailed_report(self):
        """Save detailed analysis report"""
        with open('coverage_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.coverage_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed report saved to: coverage_analysis_report.json")
    
    def run_complete_audit(self):
        """Run the complete audit analysis"""
        print("🔍 COMPREHENSIVE DATA COVERAGE AUDIT")
        print("="*60)
        print("Analyzing category definitions vs training data coverage...")
        print()
        
        # Load data
        self.load_data()
        
        # Run all analyses
        defined_categories, category_subcounts = self.analyze_category_definitions()
        training_category_counts, training_subcategory_counts, combo_counts = self.analyze_training_data()
        missing_categories, covered_categories = self.analyze_coverage_gaps(
            defined_categories, category_subcounts, training_category_counts
        )
        self.analyze_data_imbalance(training_category_counts)
        self.generate_subcategory_analysis()
        self.generate_recommendations()
        self.save_detailed_report()
        
        print("\n" + "="*60)
        print("✅ AUDIT COMPLETE - Check coverage_analysis_report.json for full details")
        print("="*60)

if __name__ == "__main__":
    auditor = DataCoverageAuditor()
    auditor.run_complete_audit()
