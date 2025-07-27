"""
STEP 2-3: STATE MANAGEMENT & AGENT PROCESSING - ENTITY MODELS

This file defines the core data models for our ITSM classification system.
These entities represent the hierarchical structure of our classification
system, the state management for ticket processing, and agent communication.

KEY CONCEPTS:
1. Category: Main classification category (Category in CSV)
2. Subcategory: Secondary classification (SubCategory in CSV)  
3. Classification Hierarchy: Complete tree structure of categories
4. Ticket State: Current processing state of a support ticket (ENHANCED for agents)
5. Agent Processing: State fields for Arabic processing, classification, validation
6. Entity Loading Stats: Metrics for data loading operations

DESIGN DECISIONS:
- Pydantic V2: Type validation and JSON serialization with modern best practices
- Mutable Agent State: Agents can update state as they process
- Rich Metadata: Keywords, usage statistics, confidence scores, and timestamps
- Hierarchical Structure: Parent-child relationships between categories
- Agent Communication: State-based message passing between specialized agents
- Vector Support: Integration with Qdrant for semantic search capabilities

The entity hierarchy matches our CSV structure:
Main Category (التسجيل) → Subcategories (التحقق من السجل التجاري, المرفقات, etc.)
"""

from typing import List, Dict, Set, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass
import csv
import os
from pathlib import Path


class Subcategory(BaseModel):
    """
    Represents a subcategory in our ITSM classification hierarchy.
    This corresponds to SubCategory in the CSV file.
    """
    name: str = Field(description="Subcategory name")
    description: str = Field(description="Subcategory description")
    parent_category: str = Field(description="Parent category name")
    
    # Metadata for analysis and validation
    keywords: Set[str] = Field(default_factory=set, description="Keywords extracted from name and description")
    usage_count: int = Field(default=0, description="Number of times this subcategory has been used")
    
    class Config:
        """Pydantic configuration"""
        # REMOVED frozen=True to allow proper mutation during loading
        json_encoders = {
            set: list  # Convert sets to lists for JSON serialization
        }
    
    def __hash__(self):
        """Make hashable for use in sets"""
        return hash((self.name, self.parent_category))
    
    def matches_keywords(self, text: str) -> bool:
        """Check if this subcategory's keywords match the given text"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)


class Category(BaseModel):
    """
    Represents a main category in our ITSM classification hierarchy.
    This corresponds to Category in the CSV file.
    """
    name: str = Field(description="Category name")
    description: str = Field(description="Category description")
    
    # Subcategories belonging to this category
    subcategories: Dict[str, Subcategory] = Field(default_factory=dict, description="Subcategories indexed by name")
    
    # Metadata for analysis and validation
    keywords: Set[str] = Field(default_factory=set, description="Keywords extracted from name and description")
    usage_count: int = Field(default=0, description="Number of times this category has been used")
    
    class Config:
        """Pydantic configuration"""
        # REMOVED frozen=True to allow proper mutation during loading
        json_encoders = {
            set: list  # Convert sets to lists for JSON serialization
        }
    
    def __hash__(self):
        """Make hashable for use in sets"""
        return hash(self.name)
    
    def add_subcategory(self, subcategory: Subcategory) -> None:
        """Add a subcategory to this category"""
        # Now safe to modify since not frozen
        self.subcategories[subcategory.name] = subcategory
    
    def get_subcategory(self, name: str) -> Optional[Subcategory]:
        """Get a subcategory by name"""
        return self.subcategories.get(name)
    
    def get_all_subcategories(self) -> List[Subcategory]:
        """Get all subcategories as a list"""
        return list(self.subcategories.values())
    
    def matches_keywords(self, text: str) -> bool:
        """Check if this category's keywords match the given text"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)
    
    def find_matching_subcategories(self, text: str) -> List[Subcategory]:
        """Find subcategories that match the given text"""
        return [sub for sub in self.subcategories.values() if sub.matches_keywords(text)]


class ClassificationHierarchy(BaseModel):
    """
    Represents the complete classification hierarchy loaded from CSV.
    This is the main entity that holds all categories and subcategories.
    """
    categories: Dict[str, Category] = Field(default_factory=dict, description="Categories indexed by name")
    total_categories: int = Field(default=0, description="Total number of categories")
    total_subcategories: int = Field(default=0, description="Total number of subcategories")
    loaded_at: Optional[str] = Field(None, description="Timestamp when hierarchy was loaded")
    source_file: Optional[str] = Field(None, description="Source CSV file path")
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            set: list  # Convert sets to lists for JSON serialization
        }
    
    def add_category(self, category: Category) -> None:
        """Add a category to the hierarchy"""
        self.categories[category.name] = category
        self._update_counts()
    
    def get_category(self, name: str) -> Optional[Category]:
        """Get a category by name"""
        return self.categories.get(name)
    
    def get_all_categories(self) -> List[Category]:
        """Get all categories as a list"""
        return list(self.categories.values())
    
    def get_all_category_names(self) -> List[str]:
        """Get all category names"""
        return list(self.categories.keys())
    
    def get_all_subcategory_names(self) -> List[str]:
        """Get all subcategory names across all categories"""
        subcategories = []
        for category in self.categories.values():
            subcategories.extend(category.subcategories.keys())
        return subcategories
    
    def find_category_by_subcategory(self, subcategory_name: str) -> Optional[Category]:
        """Find the category that contains a specific subcategory"""
        for category in self.categories.values():
            if subcategory_name in category.subcategories:
                return category
        return None
    
    def validate_classification(self, main_category: str, subcategory: str) -> bool:
        """Validate that a classification combination exists in the hierarchy"""
        category = self.get_category(main_category)
        if not category:
            return False
        return subcategory in category.subcategories
    
    def search_categories(self, query: str) -> List[Category]:
        """Search for categories that match the query text"""
        return [cat for cat in self.categories.values() if cat.matches_keywords(query)]
    
    def search_subcategories(self, query: str) -> List[tuple[Category, Subcategory]]:
        """Search for subcategories that match the query text"""
        results = []
        for category in self.categories.values():
            matching_subs = category.find_matching_subcategories(query)
            for sub in matching_subs:
                results.append((category, sub))
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the hierarchy"""
        subcategory_counts = [len(cat.subcategories) for cat in self.categories.values()]
        return {
            "total_categories": self.total_categories,
            "total_subcategories": self.total_subcategories,
            "avg_subcategories_per_category": sum(subcategory_counts) / len(subcategory_counts) if subcategory_counts else 0,
            "max_subcategories_per_category": max(subcategory_counts) if subcategory_counts else 0,
            "min_subcategories_per_category": min(subcategory_counts) if subcategory_counts else 0,
            "categories_with_no_subcategories": sum(1 for count in subcategory_counts if count == 0)
        }
    
    def _update_counts(self) -> None:
        """Update the total counts"""
        self.total_categories = len(self.categories)
        self.total_subcategories = sum(len(cat.subcategories) for cat in self.categories.values())
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ClassificationHierarchy':
        """Load hierarchical classification structure from CSV file"""
        import csv
        from datetime import datetime
        
        hierarchy = cls()
        hierarchy.source_file = file_path
        hierarchy.loaded_at = datetime.now().isoformat()
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: Hierarchy file not found: {file_path}")
            return hierarchy
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                categories_created = 0
                subcategories_created = 0
                current_category = None
                current_category_name = None
                
                print(f"🔄 Loading hierarchical data from {file_path}...")
                
                for row_num, row in enumerate(csv_reader, 1):
                    category_name = row.get('Category', '').strip()
                    category_desc = row.get('Category_Description', '').strip()
                    subcategory_name = row.get('SubCategory', '').strip()
                    subcategory_desc = row.get('SubCategory_Description', '').strip()
                    
                    # If we have a category name, this is a new category definition
                    if category_name:
                        current_category_name = category_name
                        
                        # Create new category
                        if category_name not in hierarchy.categories:
                            current_category = Category(
                                name=category_name,
                                description=category_desc or f"Category: {category_name}",
                                keywords=hierarchy._extract_keywords(category_name + " " + category_desc)
                            )
                            hierarchy.categories[category_name] = current_category
                            categories_created += 1
                            print(f"  📁 Created category: {category_name}")
                        else:
                            current_category = hierarchy.categories[category_name]
                    
                    # Add subcategory to the current category (if we have one)
                    if subcategory_name and current_category and current_category_name:
                        if subcategory_name not in current_category.subcategories:
                            subcategory = Subcategory(
                                name=subcategory_name,
                                description=subcategory_desc or f"Subcategory: {subcategory_name}",
                                parent_category=current_category_name,
                                keywords=hierarchy._extract_keywords(subcategory_name + " " + subcategory_desc)
                            )
                            current_category.subcategories[subcategory_name] = subcategory
                            subcategories_created += 1
                            print(f"    ➕ Added subcategory: {subcategory_name}")
                    
                    # Skip rows with no meaningful data
                    elif not category_name and not subcategory_name:
                        continue
            
            # Update final counts
            hierarchy._update_counts()
            
            print(f"✅ Hierarchical structure loaded successfully!")
            print(f"📊 Summary: {categories_created} categories, {subcategories_created} subcategories")
            print(f"🏗️ Total structure: {hierarchy.total_categories} categories, {hierarchy.total_subcategories} subcategories")
            
            # Show category breakdown
            if hierarchy.total_categories > 0:
                print(f"📋 Category breakdown:")
                for cat_name, category in list(hierarchy.categories.items())[:5]:
                    print(f"  • {cat_name}: {len(category.subcategories)} subcategories")
                if hierarchy.total_categories > 5:
                    print(f"  ... and {hierarchy.total_categories - 5} more categories")
            
        except Exception as e:
            print(f"❌ Error loading hierarchical structure from {file_path}: {e}")
            import traceback
            traceback.print_exc()
        
        return hierarchy
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text for matching purposes with proper Arabic handling"""
        if not text or text.strip() == "":
            return set()
        
        import re
        
        # Remove Arabic diacritics first
        arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        text = arabic_diacritics.sub('', text)
        
        # Split on Arabic and English punctuation
        words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z0-9]+', text)
        
        # Filter and normalize
        keywords = set()
        for word in words:
            word = word.strip()
            if len(word) > 2:  # Minimum length
                keywords.add(word)
                # Add normalized form if different (handle different Arabic alif forms)
                normalized = word.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
                if normalized != word:
                    keywords.add(normalized)
        
        return keywords


@dataclass
class EntityLoadingStats:
    """Statistics about entity loading process"""
    total_rows_processed: int = 0
    categories_created: int = 0
    subcategories_created: int = 0
    empty_rows_skipped: int = 0
    errors_encountered: int = 0
    loading_time_ms: int = 0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# Type aliases for convenience
CategoryDict = Dict[str, Category]
SubcategoryDict = Dict[str, Subcategory]
SearchResult = List[tuple[Category, Subcategory]]
