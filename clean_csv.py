#!/usr/bin/env python3
"""
CSV Cleaning Script
Removes empty rows from Thiqa_Incidents_Example.csv and keeps only valid incidents
"""

import csv
import os
from datetime import datetime

def clean_csv_file():
    """Clean the CSV file by removing empty rows"""
    
    input_file = "Thiqa_Incidents_Example.csv"
    output_file = "Thiqa_Incidents_Example_cleaned.csv"
    backup_file = f"Thiqa_Incidents_Example_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"🧹 CSV Cleaning Script")
    print("=" * 50)
    print(f"📂 Input file: {input_file}")
    print(f"💾 Backup file: {backup_file}")
    print(f"✨ Output file: {output_file}")
    print()
    
    try:
        # First, create a backup
        print("📋 Creating backup...")
        os.rename(input_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
        
        # Read the backup file
        print("📖 Reading CSV data...")
        with open(backup_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            all_rows = list(reader)
        
        print(f"📊 Total rows in original file: {len(all_rows)}")
        
        # Filter out empty rows
        valid_rows = []
        for row in all_rows:
            # Check if Description has actual content
            description = row.get('Description', '').strip()
            if description:
                valid_rows.append(row)
        
        print(f"✅ Valid rows with content: {len(valid_rows)}")
        print(f"🗑️  Empty rows removed: {len(all_rows) - len(valid_rows)}")
        
        # Write cleaned data to new file
        print("✍️  Writing cleaned data...")
        if valid_rows:
            # Get the field names from the first valid row
            fieldnames = list(valid_rows[0].keys())
            
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(valid_rows)
            
            print(f"✅ Cleaned file written: {output_file}")
            
            # Replace the original file with cleaned version
            os.rename(output_file, input_file)
            print(f"🔄 Original file replaced with cleaned version")
            
            # Show sample of cleaned data
            print("\n📋 Sample of cleaned data:")
            print("-" * 80)
            for i, row in enumerate(valid_rows[:3], 1):
                incident_key = next((k for k in row.keys() if 'Incident' in k), 'Incident')
                incident_id = row[incident_key]
                description = row['Description'][:100] + "..." if len(row['Description']) > 100 else row['Description']
                category = row.get('Subcategory_Thiqah', 'N/A')
                subcategory = row.get('Subcategory2_Thiqah', 'N/A')
                
                print(f"Row {i}:")
                print(f"  ID: {incident_id}")
                print(f"  Description: {description}")
                print(f"  Category: {category}")
                print(f"  Subcategory: {subcategory}")
                print()
        
        else:
            print("❌ No valid rows found!")
            return False
        
        print("🎉 CSV cleaning completed successfully!")
        print(f"📁 Backup available at: {backup_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        # Try to restore backup if something went wrong
        if os.path.exists(backup_file) and not os.path.exists(input_file):
            os.rename(backup_file, input_file)
            print("🔄 Backup restored due to error")
        return False

if __name__ == "__main__":
    success = clean_csv_file()
    if success:
        print("\n✅ Ready to run demo with cleaned data!")
    else:
        print("\n❌ Cleaning failed. Please check the error messages.")
