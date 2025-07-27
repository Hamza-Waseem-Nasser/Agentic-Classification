"""
Classification Validator - Strict Validation for Category and Subcategory Classifications

This utility ensures that all classifications adhere strictly to the predefined hierarchy
from the CSV file. It prevents fuzzy matching and similar name acceptance.
"""

from typing import List, Tuple, Optional, Set, Dict
from ..models.entities import ClassificationHierarchy


class ClassificationValidator:
    """Strict validator for category and subcategory classifications"""
    
    def __init__(self, hierarchy: ClassificationHierarchy):
        self.hierarchy = hierarchy
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast validation"""
        self.valid_categories = set(self.hierarchy.categories.keys())
        self.category_subcategory_map = {}
        
        for cat_name, category in self.hierarchy.categories.items():
            self.category_subcategory_map[cat_name] = set(category.subcategories.keys())
    
    def validate_category(self, category: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if category exists exactly in hierarchy
        Returns: (is_valid, error_message)
        """
        if category in self.valid_categories:
            return True, None
        
        return False, f"Category '{category}' is not valid. Valid categories are: {', '.join(sorted(self.valid_categories))}"
    
    def validate_subcategory(self, category: str, subcategory: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if subcategory exists under the given category
        Returns: (is_valid, error_message)
        """
        if category not in self.valid_categories:
            return False, f"Parent category '{category}' is not valid"
        
        valid_subcategories = self.category_subcategory_map.get(category, set())
        
        if subcategory in valid_subcategories:
            return True, None
            
        return False, f"Subcategory '{subcategory}' is not valid for category '{category}'. Valid subcategories are: {', '.join(sorted(valid_subcategories))}"
    
    def get_valid_categories(self) -> List[str]:
        """Get list of valid category names"""
        return list(self.valid_categories)
    
    def get_valid_subcategories_for_category(self, category: str) -> List[str]:
        """Get list of valid subcategories for a specific category"""
        if category in self.category_subcategory_map:
            return list(self.category_subcategory_map[category])
        return []
    
    def is_valid_classification(self, category: str, subcategory: str) -> bool:
        """Check if a category/subcategory combination is valid"""
        if category not in self.valid_categories:
            return False
            
        return subcategory in self.category_subcategory_map.get(category, set())
    
    def get_valid_categories_prompt(self) -> str:
        """Get a formatted string of valid categories for LLM prompts"""
        return "\n".join([f"- {cat}" for cat in sorted(self.valid_categories)])
    
    def get_valid_subcategories_prompt(self, category: str) -> str:
        """Get a formatted string of valid subcategories for a category"""
        if category in self.category_subcategory_map:
            subcats = self.category_subcategory_map[category]
            return "\n".join([f"- {subcat}" for subcat in sorted(subcats)])
        return "No valid subcategories found for this category"
    
    def get_default_category(self) -> str:
        """Get a default category (first valid category)"""
        if self.valid_categories:
            return sorted(self.valid_categories)[0]
        return "غير محدد"
    
    def find_closest_valid_category(self, invalid_category: str) -> Optional[str]:
        """
        Find the closest valid category (for fallback purposes only)
        This should be used sparingly and with reduced confidence
        """
        invalid_lower = invalid_category.lower().strip()
        
        # First try exact case-insensitive match
        for valid_cat in self.valid_categories:
            if valid_cat.lower().strip() == invalid_lower:
                return valid_cat
        
        # Then try substring match (very conservative)
        for valid_cat in self.valid_categories:
            if len(invalid_lower) > 3 and (
                invalid_lower in valid_cat.lower() or 
                valid_cat.lower() in invalid_lower
            ):
                return valid_cat
        
        # No match found
        return None
