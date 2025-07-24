"""
STEP 3: SUBCATEGORY CLASSIFIER AGENT - HIERARCHICAL CLASSIFICATION

This agent performs the second level of classification, determining the specific
subcategory within the main category using context-aware classification and
hierarchical reasoning. It leverages the main category context to narrow down options.

KEY RESPONSIBILITIES:
1. Context-Aware Classification: Use main category to constrain subcategory options
2. Hierarchical Reasoning: Understand parent-child relationships in classification
3. Semantic Matching: Use vector search within category constraints
4. Confidence Assessment: Provide reliable confidence scores for subcategory selection
5. Fallback Handling: Graceful degradation when no clear subcategory match exists

CLASSIFICATION PIPELINE:
Main Category + Processed Text → Filtered Subcategories → Vector Search → LLM Classification → Final Subcategory

DESIGN DECISIONS:
- Hierarchical Approach: Only consider subcategories under the identified main category
- Context Enhancement: Use main category information to improve classification accuracy
- Vector Filtering: Search only within relevant subcategory embeddings
- Confidence Calibration: Adjust confidence based on main category confidence
- Fallback Strategy: Default to "عام" subcategory when no specific match found

HIERARCHICAL STRATEGY:
- Load subcategories for the identified main category only
- Use category-specific embeddings and examples for better matching
- Apply category-specific business rules and thresholds
- Leverage parent category context in LLM prompts for better reasoning

INTEGRATION POINTS:
- CategoryClassifierAgent: Uses main category result as input constraint
- Qdrant Vector Database: Filtered search within category-specific embeddings
- Classification Hierarchy: Direct access to parent-child relationships
- Business Rules: Category-specific classification rules and thresholds
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# We'll import Qdrant dependencies when needed to avoid import errors
QDRANT_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    Filter = FieldCondition = MatchValue = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None

from .base_agent import BaseAgent, BaseAgentConfig, AgentType
from ..models.ticket_state import TicketState
from ..models.entities import Category, Subcategory, ClassificationHierarchy


class SubcategoryClassifierAgent(BaseAgent):
    """
    Subcategory Classifier Agent: Context-aware hierarchical classification.
    
    Performs precise subcategory classification by leveraging the main category
    context to narrow down options and improve classification accuracy.
    """
    
    def __init__(self, config: BaseAgentConfig, 
                 hierarchy: Optional[ClassificationHierarchy] = None,
                 qdrant_client=None,
                 collection_name: str = "itsm_categories"):
        super().__init__(config)
        
        self.hierarchy = hierarchy
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AsyncOpenAI()
        
        # Classification parameters
        self.embedding_model = "text-embedding-3-small"
        self.top_k_subcategories = 3
        self.minimum_confidence = 0.6
        
        # Category-specific configurations
        self.category_configs = self._load_category_configs()
        
        # Default subcategory for fallback
        self.default_subcategory = "عام"
        
        self.logger.info("Subcategory Classifier initialized with hierarchical reasoning")
    
    def _load_category_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load category-specific configuration and business rules"""
        return {
            # Login and Authentication
            "تسجيل الدخول": {
                "confidence_threshold": 0.7,
                "keywords_weight": 0.3,
                "common_keywords": ["دخول", "تسجيل", "كلمة مرور", "حساب"],
                "fallback_subcategory": "عدم القدرة على تسجيل الدخول"
            },
            
            # Payments and Financial
            "المدفوعات": {
                "confidence_threshold": 0.8,  # Higher threshold for financial
                "keywords_weight": 0.4,
                "common_keywords": ["دفع", "فاتورة", "مبلغ", "بنك"],
                "fallback_subcategory": "مشاكل في عملية الدفع"
            },
            
            # Registration and Setup
            "التسجيل": {
                "confidence_threshold": 0.7,
                "keywords_weight": 0.3,
                "common_keywords": ["تسجيل", "إنشاء", "حساب", "مستخدم"],
                "fallback_subcategory": "مشاكل في التسجيل"
            },
            
            # Products and Inventory
            "إضافة المنتجات": {
                "confidence_threshold": 0.75,
                "keywords_weight": 0.35,
                "common_keywords": ["منتج", "إضافة", "بضاعة", "مخزون"],
                "fallback_subcategory": "مشاكل في إضافة المنتجات"
            },
            
            # Default configuration for other categories
            "default": {
                "confidence_threshold": 0.65,
                "keywords_weight": 0.25,
                "common_keywords": [],
                "fallback_subcategory": "عام"
            }
        }
    
    async def process(self, state: TicketState) -> TicketState:
        """
        Classify ticket into specific subcategory within the main category.
        
        Args:
            state: Ticket state with main category classification
            
        Returns:
            Enhanced state with subcategory classification
        """
        self.logger.info(f"Starting subcategory classification for ticket {state.ticket_id}")
        
        # Validate that main category classification exists
        if not hasattr(state, 'classification') or not state.classification.main_category:
            raise ValueError("Main category classification required for subcategory classification")
        
        main_category = state.classification.main_category
        processed_text = getattr(state, 'processed_text', None) or state.original_text
        
        # 1. Get available subcategories for the main category
        available_subcategories = await self._get_category_subcategories(main_category)
        
        if not available_subcategories:
            # No subcategories available, use default
            await self._set_default_subcategory(state, main_category)
            return state
        
        # 2. Find most relevant subcategories using vector search
        relevant_subcategories = await self._find_relevant_subcategories(
            processed_text, main_category, available_subcategories
        )
        
        # 3. Classify using LLM with hierarchical context
        subcategory_result = await self._classify_subcategory_with_llm(
            processed_text, main_category, relevant_subcategories
        )
        
        # 4. Validate and store results
        await self._validate_and_store_subcategory(state, subcategory_result, available_subcategories)
        
        self.logger.info(f"Subcategory classification complete: {subcategory_result.get('subcategory', 'unknown')}")
        return state
    
    async def _get_category_subcategories(self, main_category: str) -> List[Dict[str, Any]]:
        """Get all available subcategories for the main category"""
        
        if not self.hierarchy or main_category not in self.hierarchy.categories:
            self.logger.warning(f"No subcategories found for category: {main_category}")
            return []
        
        category = self.hierarchy.categories[main_category]
        subcategories = []
        
        for subcategory_name, subcategory in category.subcategories.items():
            subcategories.append({
                "name": subcategory_name,
                "description": subcategory.description,
                "keywords": list(subcategory.keywords),
                "parent_category": main_category,
                "usage_count": subcategory.usage_count
            })
        
        self.logger.debug(f"Found {len(subcategories)} subcategories for '{main_category}'")
        return subcategories
    
    async def _find_relevant_subcategories(self, text: str, main_category: str, 
                                         available_subcategories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find most relevant subcategories using vector search within category"""
        
        if not self.qdrant_client:
            # Fallback to keyword matching if no vector search available
            return self._keyword_based_subcategory_matching(text, available_subcategories)
        
        try:
            # Get embedding for the input text
            from .category_classifier_agent import CategoryClassifierAgent
            query_embedding = await self._get_embedding(text)
            
            if not query_embedding:
                return self._keyword_based_subcategory_matching(text, available_subcategories)
            
            # Create filter to search only within the main category
            if not QDRANT_AVAILABLE:
                return self._keyword_based_subcategory_matching(text, available_subcategories)
            
            # Use proper Qdrant filtering for category-specific search
            category_filter = Filter(
                must=[
                    FieldCondition(
                        key="category_name",
                        match=MatchValue(value=main_category)
                    ),
                    FieldCondition(
                        key="type", 
                        match=MatchValue(value="subcategory_context")
                    )
                ]
            )
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=category_filter,
                limit=self.top_k_subcategories * 2,
                score_threshold=0.6
            )
            
            # Process results and merge with available subcategories
            relevant_subcategories = []
            found_subcategories = set()
            
            for result in search_results:
                if "subcategory_name" in result.payload:
                    subcategory_name = result.payload["subcategory_name"]
                    
                    if subcategory_name not in found_subcategories:
                        # Find matching subcategory in available list
                        for sub in available_subcategories:
                            if sub["name"] == subcategory_name:
                                sub["similarity_score"] = result.score
                                relevant_subcategories.append(sub)
                                found_subcategories.add(subcategory_name)
                                break
            
            # Add remaining subcategories with lower scores if we don't have enough
            if len(relevant_subcategories) < self.top_k_subcategories:
                for sub in available_subcategories:
                    if sub["name"] not in found_subcategories:
                        sub["similarity_score"] = 0.3  # Low baseline score
                        relevant_subcategories.append(sub)
                        
                        if len(relevant_subcategories) >= self.top_k_subcategories:
                            break
            
            # Sort by similarity score
            relevant_subcategories.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            self.logger.debug(f"Found {len(relevant_subcategories)} relevant subcategories via vector search")
            return relevant_subcategories[:self.top_k_subcategories]
            
        except Exception as e:
            self.logger.warning(f"Vector search failed for subcategories: {e}")
            return self._keyword_based_subcategory_matching(text, available_subcategories)
    
    def _keyword_based_subcategory_matching(self, text: str, 
                                          available_subcategories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback keyword-based subcategory matching"""
        
        text_lower = text.lower()
        scored_subcategories = []
        
        for subcategory in available_subcategories:
            score = 0
            
            # Score based on name matching
            if subcategory["name"].lower() in text_lower:
                score += 0.5
            
            # Score based on description matching
            if subcategory["description"] and subcategory["description"].lower() in text_lower:
                score += 0.3
            
            # Score based on keyword matching
            keyword_matches = sum(1 for keyword in subcategory["keywords"] 
                                if keyword.lower() in text_lower)
            if subcategory["keywords"]:
                score += (keyword_matches / len(subcategory["keywords"])) * 0.4
            
            subcategory["similarity_score"] = score
            scored_subcategories.append(subcategory)
        
        # Sort by score and return top candidates
        scored_subcategories.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        self.logger.debug(f"Keyword-based matching found {len(scored_subcategories)} scored subcategories")
        return scored_subcategories[:self.top_k_subcategories]
    
    async def _classify_subcategory_with_llm(self, text: str, main_category: str, 
                                           relevant_subcategories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify subcategory using LLM with hierarchical context"""
        
        if not self.llm:
            # Fallback to highest scoring subcategory
            if relevant_subcategories:
                best_sub = relevant_subcategories[0]
                return {
                    "subcategory": best_sub["name"],
                    "confidence": best_sub.get("similarity_score", 0.5),
                    "reasoning": "Fallback to highest similarity match (no LLM)",
                    "classification_method": "similarity_fallback"
                }
            else:
                return {
                    "subcategory": self.default_subcategory,
                    "confidence": 0.3,
                    "reasoning": "No relevant subcategories found",
                    "classification_method": "default_fallback"
                }
        
        # Get category-specific configuration
        category_config = self.category_configs.get(main_category, self.category_configs["default"])
        
        # Build hierarchical classification prompt
        prompt = self._build_subcategory_prompt(text, main_category, relevant_subcategories, category_config)
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_message = SystemMessage(content=f"""
            أنت خبير في التصنيف الهرمي للفئات الفرعية في نظام إدارة تذاكر الدعم التقني.
            المهمة: تصنيف النص إلى فئة فرعية محددة ضمن الفئة الرئيسية "{main_category}".
            
            قواعد التصنيف:
            1. اختر الفئة الفرعية الأكثر دقة وتخصصاً
            2. استخدم السياق والكلمات المفتاحية للتحديد
            3. في حالة عدم وجود تطابق واضح، اختر الفئة الأعم
            4. أعط درجة ثقة من 0 إلى 1
            
            أجب بصيغة JSON فقط:
            {{"subcategory": "اسم الفئة الفرعية", "confidence": 0.95, "reasoning": "سبب الاختيار"}}
            """)
            
            human_message = HumanMessage(content=prompt)
            
            response = await self._safe_llm_call([system_message, human_message])
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                
                # Validate required fields
                if "subcategory" not in result:
                    raise ValueError("Missing subcategory in response")
                
                # Ensure confidence is a float
                result["confidence"] = float(result.get("confidence", 0.5))
                
                # Adjust confidence based on main category confidence
                if hasattr(self, '_adjust_confidence_based_on_parent'):
                    result["confidence"] = self._adjust_confidence_based_on_parent(
                        result["confidence"], main_category
                    )
                
                # Add metadata
                result["classification_method"] = "llm_hierarchical"
                result["parent_category"] = main_category
                result["relevant_subcategories_used"] = len(relevant_subcategories)
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM subcategory response as JSON: {e}")
                
                # Fallback: extract subcategory from text response
                return self._extract_subcategory_from_text(response, relevant_subcategories, main_category)
                
        except Exception as e:
            self.logger.error(f"LLM subcategory classification failed: {e}")
            
            # Fallback to highest scoring subcategory
            if relevant_subcategories:
                best_sub = relevant_subcategories[0]
                return {
                    "subcategory": best_sub["name"],
                    "confidence": best_sub.get("similarity_score", 0.4),
                    "reasoning": f"LLM failed, using highest similarity match: {str(e)[:100]}",
                    "classification_method": "similarity_fallback",
                    "parent_category": main_category
                }
            else:
                return {
                    "subcategory": category_config["fallback_subcategory"],
                    "confidence": 0.3,
                    "reasoning": "LLM failed and no relevant subcategories found",
                    "classification_method": "category_default_fallback",
                    "parent_category": main_category
                }
    
    def _build_subcategory_prompt(self, text: str, main_category: str, 
                                relevant_subcategories: List[Dict[str, Any]], 
                                category_config: Dict[str, Any]) -> str:
        """Build hierarchical subcategory classification prompt"""
        
        prompt_parts = [
            f"قم بتصنيف النص التالي إلى فئة فرعية ضمن الفئة الرئيسية '{main_category}':",
            f"النص: {text}",
            "",
            "الفئات الفرعية المتاحة (مرتبة حسب الصلة):"
        ]
        
        for i, subcategory in enumerate(relevant_subcategories, 1):
            similarity_info = ""
            if "similarity_score" in subcategory:
                similarity_info = f" (تشابه: {subcategory['similarity_score']:.2f})"
            
            prompt_parts.append(
                f"{i}. {subcategory['name']}: {subcategory['description']}{similarity_info}"
            )
            
            # Add keywords if available
            if subcategory.get("keywords"):
                keywords_str = "، ".join(subcategory["keywords"][:5])
                prompt_parts.append(f"   الكلمات المفتاحية: {keywords_str}")
        
        prompt_parts.extend([
            "",
            f"متطلبات الجودة لهذه الفئة:",
            f"- الحد الأدنى للثقة: {category_config['confidence_threshold']}",
            f"- وزن الكلمات المفتاحية: {category_config['keywords_weight']}",
            ""
        ])
        
        # Add category-specific guidance
        if category_config.get("common_keywords"):
            keywords = "، ".join(category_config["common_keywords"])
            prompt_parts.append(f"الكلمات المهمة في هذه الفئة: {keywords}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "اختر الفئة الفرعية الأكثر دقة وتخصصاً مع درجة الثقة ومبرر الاختيار.",
            "أجب بصيغة JSON فقط:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_subcategory_from_text(self, response: str, 
                                     relevant_subcategories: List[Dict[str, Any]], 
                                     main_category: str) -> Dict[str, Any]:
        """Extract subcategory from text response as fallback"""
        
        response_lower = response.lower()
        
        # Look for subcategory names in the response
        for subcategory in relevant_subcategories:
            if subcategory["name"].lower() in response_lower:
                return {
                    "subcategory": subcategory["name"],
                    "confidence": 0.6,  # Medium confidence for text extraction
                    "reasoning": f"Extracted from response: {response[:100]}...",
                    "classification_method": "text_extraction_fallback",
                    "parent_category": main_category
                }
        
        # If no match found, use first relevant subcategory
        if relevant_subcategories:
            return {
                "subcategory": relevant_subcategories[0]["name"],
                "confidence": 0.4,
                "reasoning": "Fallback to most relevant subcategory",
                "classification_method": "relevance_fallback",
                "parent_category": main_category
            }
        
        # Final fallback to category default
        category_config = self.category_configs.get(main_category, self.category_configs["default"])
        return {
            "subcategory": category_config["fallback_subcategory"],
            "confidence": 0.2,
            "reasoning": "No subcategories found in response",
            "classification_method": "category_default_fallback",
            "parent_category": main_category
        }
    
    async def _set_default_subcategory(self, state: TicketState, main_category: str):
        """Set default subcategory when no subcategories are available"""
        
        category_config = self.category_configs.get(main_category, self.category_configs["default"])
        
        if not hasattr(state, 'classification'):
            state.classification = {}
        
        state.classification.update({
            "subcategory": category_config["fallback_subcategory"],
            "subcategory_confidence": 0.5,
            "subcategory_reasoning": f"No subcategories available for '{main_category}', using default",
            "classification_method": "no_subcategories_default"
        })
        
        self.logger.info(f"Set default subcategory '{category_config['fallback_subcategory']}' for '{main_category}'")
    
    async def _validate_and_store_subcategory(self, state: TicketState, 
                                            subcategory_result: Dict[str, Any],
                                            available_subcategories: List[Dict[str, Any]]):
        """Validate subcategory result and store in state"""
        
        subcategory = subcategory_result.get("subcategory", "")
        confidence = subcategory_result.get("confidence", 0.0)
        reasoning = subcategory_result.get("reasoning", "")
        main_category = subcategory_result.get("parent_category", "")
        
        # Validate subcategory exists in available options
        valid_subcategory_names = [sub["name"] for sub in available_subcategories]
        
        if subcategory not in valid_subcategory_names and available_subcategories:
            self.logger.warning(f"Subcategory '{subcategory}' not in available options")
            
            # Try to find a close match
            best_match = self._find_closest_subcategory_match(subcategory, available_subcategories)
            if best_match:
                subcategory = best_match["name"]
                confidence *= 0.7  # Reduce confidence for fallback
                reasoning += f" (adjusted to closest match: {best_match['name']})"
        
        # Store in ticket state
        state.predicted_subcategory = subcategory
        state.subcategory_confidence = confidence
        
        # Store in classification object
        if not hasattr(state.classification, 'subcategory'):
            # Initialize subcategory attributes if they don't exist
            state.classification.subcategory = None
            state.classification.subcategory_description = None
        
        state.classification.subcategory = subcategory
        state.classification.confidence_score = max(state.classification.confidence_score, confidence)
        
        # Store subcategory metadata
        if not hasattr(state, 'classification_metadata'):
            state.classification_metadata = {}
        
        state.classification_metadata['subcategory_agent'] = {
            'processing_timestamp': datetime.now().isoformat(),
            'available_subcategories': len(available_subcategories),
            'classification_method': subcategory_result.get("classification_method"),
            'parent_category': main_category,
            'confidence_threshold_met': confidence >= self.minimum_confidence
        }
        
        # Update overall review requirement
        if hasattr(state.classification_metadata, 'requires_review'):
            state.classification_metadata['requires_review'] = (
                state.classification_metadata.get('requires_review', False) or 
                confidence < self.minimum_confidence
            )
        
        self.logger.debug(f"Stored subcategory: {subcategory} (confidence: {confidence:.2f})")
    
    def _find_closest_subcategory_match(self, target: str, 
                                      available_subcategories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest matching subcategory name"""
        
        target_lower = target.lower()
        best_match = None
        best_score = 0
        
        for subcategory in available_subcategories:
            # Simple substring matching
            name_lower = subcategory["name"].lower()
            
            if target_lower in name_lower or name_lower in target_lower:
                score = len(set(target_lower.split()) & set(name_lower.split()))
                if score > best_score:
                    best_score = score
                    best_match = subcategory
        
        return best_match
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get OpenAI embedding for text (shared with category classifier)"""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.warning(f"Failed to get embedding for text: {e}")
            return None
    
    def _validate_output_state(self, state: TicketState) -> None:
        """Validate subcategory classification results"""
        super()._validate_output_state(state)
        
        # Subcategory classification specific validations
        if not hasattr(state, 'classification') or not state.classification.subcategory:
            raise ValueError("Subcategory classification not completed")
        
        if not isinstance(getattr(state.classification, 'confidence_score', 0), (int, float)):
            raise ValueError("Subcategory confidence score not set properly")
    
    async def get_subcategory_stats(self) -> Dict[str, Any]:
        """Get comprehensive subcategory classification statistics"""
        
        stats = {
            'agent_metrics': self.metrics.dict(),
            'classification_config': {
                'embedding_model': self.embedding_model,
                'top_k_subcategories': self.top_k_subcategories,
                'minimum_confidence': self.minimum_confidence,
                'default_subcategory': self.default_subcategory
            },
            'category_configs': self.category_configs,
            'hierarchy_status': {
                'hierarchy_loaded': self.hierarchy is not None,
                'total_categories': len(self.hierarchy.categories) if self.hierarchy else 0,
                'vector_search_available': self.qdrant_client is not None
            }
        }
        
        return stats
