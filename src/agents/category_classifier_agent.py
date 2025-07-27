"""
STEP 3: CATEGORY CLASSIFIER AGENT - MAIN CLASSIFICATION WITH VECTOR SEARCH

This agent performs the first level of classification, determining the main
category for each ticket using a combination of semantic search and LLM
classification. It leverages Qdrant vector database for efficient similarity matching.

KEY RESPONSIBILITIES:
1. Vector-Based Similarity Search: Use embeddings to find similar categories
2. LLM Classification: Use GPT-4 with few-shot examples for final decision
3. Confidence Scoring: Generate reliable confidence scores for classifications
4. Few-Shot Learning: Maintain and use relevant classification examples
5. Category Validation: Ensure classifications match the hierarchy

CLASSIFICATION PIPELINE:
Processed Text → Embeddings → Vector Search → Top-K Categories → LLM + Few-Shot → Final Classification

DESIGN DECISIONS:
- Hybrid Approach: Combines vector similarity with LLM reasoning
- Qdrant Integration: Production-ready vector database for semantic search
- OpenAI Embeddings: High-quality text embeddings for Arabic text
- Few-Shot Learning: Dynamic example selection for improved accuracy
- Confidence Thresholds: Configurable confidence levels for quality control

VECTOR SEARCH STRATEGY:
- Embed category descriptions and examples in Qdrant
- Query with processed ticket text to find semantically similar categories
- Use top-5 candidates as context for LLM classification
- Combine embedding similarity with LLM reasoning for final decision

INTEGRATION POINTS:
- Qdrant Vector Database: Semantic search and similarity matching
- OpenAI Embeddings: Text-to-vector conversion for Arabic content
- CategoryLoader: Access to category hierarchy and metadata
- ArabicProcessingAgent: Uses processed and normalized text
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI

from .base_agent import BaseAgent, BaseAgentConfig, AgentType
from ..models.ticket_state import TicketState
from ..models.entities import Category, ClassificationHierarchy


class CategoryClassifierAgent(BaseAgent):
    """
    Category Classifier Agent: Main category classification with vector search.
    
    Combines semantic similarity search using Qdrant with LLM-based classification
    to achieve high accuracy in determining the main category for each ticket.
    """
    
    def __init__(self, config: BaseAgentConfig, 
                 hierarchy: Optional[ClassificationHierarchy] = None,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "itsm_categories"):
        super().__init__(config)
        
        self.hierarchy = hierarchy
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize OpenAI client for embeddings - with API key validation
        if not config.api_key:
            raise ValueError("OpenAI API key is required for embeddings")
        self.openai_client = AsyncOpenAI(api_key=config.api_key)
        
        # Classification parameters
        self.embedding_model = "text-embedding-3-small"
        self.top_k_candidates = 5
        self.similarity_threshold = 0.5
        
        # Few-shot examples cache
        self.few_shot_cache = {}
        
        self.logger.info(f"Category Classifier initialized with Qdrant at {qdrant_url}")
        
        # Flag to track initialization status
        self._vector_collection_initialized = False
    
    @classmethod
    async def create(cls, config: BaseAgentConfig, 
                     hierarchy: Optional[ClassificationHierarchy] = None,
                     qdrant_url: str = "http://localhost:6333",
                     collection_name: str = "itsm_categories"):
        """
        Async factory method to properly initialize CategoryClassifierAgent.
        
        Args:
            config: Agent configuration
            hierarchy: Classification hierarchy
            qdrant_url: Qdrant server URL
            collection_name: Collection name for vectors
            
        Returns:
            Fully initialized CategoryClassifierAgent
        """
        instance = cls(config, hierarchy, qdrant_url, collection_name)
        await instance._initialize_vector_collection()
        return instance
    
    async def _initialize_vector_collection(self):
        """Initialize Qdrant collection for category vectors"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                # Create collection with appropriate vector size (1536 for text-embedding-3-small)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                self.logger.info(f"Created Qdrant collection: {self.collection_name}")
                
                # Initialize with category data if hierarchy is available
                if self.hierarchy:
                    await self._populate_vector_collection()
            else:
                self.logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            
            self._vector_collection_initialized = True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant collection: {e}")
            self._vector_collection_initialized = False
    
    async def _populate_vector_collection(self):
        """Populate Qdrant collection with category embeddings"""
        try:
            points = []
            point_id = 0
            
            for category_name, category in self.hierarchy.categories.items():
                # Create embeddings for category name and description
                category_text = f"{category.name} {category.description}"
                
                # Get embedding
                embedding = await self._get_embedding(category_text)
                
                if embedding:
                    # Create point for Qdrant
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "category_name": category.name,
                            "category_description": category.description,
                            "subcategory_count": len(category.subcategories),
                            "keywords": list(category.keywords),
                            "type": "main_category"
                        }
                    )
                    points.append(point)
                    point_id += 1
                
                # Add subcategories as additional context
                for subcategory in category.subcategories.values():
                    subcategory_text = f"{subcategory.name} {subcategory.description}"
                    subcategory_embedding = await self._get_embedding(subcategory_text)
                    
                    if subcategory_embedding:
                        point = PointStruct(
                            id=point_id,
                            vector=subcategory_embedding,
                            payload={
                                "category_name": category.name,
                                "subcategory_name": subcategory.name,
                                "subcategory_description": subcategory.description,
                                "parent_category": category.name,
                                "type": "subcategory_context"
                            }
                        )
                        points.append(point)
                        point_id += 1
            
            # Upload points to Qdrant
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.logger.info(f"Populated Qdrant with {len(points)} category embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to populate vector collection: {e}")
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get OpenAI embedding for text"""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.warning(f"Failed to get embedding for text: {e}")
            return None
    
    async def process(self, state: TicketState) -> TicketState:
        """
        Classify ticket into main category using vector search + LLM.
        
        Args:
            state: Ticket state with processed Arabic text
            
        Returns:
            Enhanced state with main category classification
        """
        self.logger.info(f"Starting category classification for ticket {state.ticket_id}")
        
        # Get processed text (should be available from Arabic agent)
        text = getattr(state, 'processed_text', None) or state.original_text
        
        if not text or not text.strip():
            raise ValueError("No text available for classification")
        
        # 1. Vector-based similarity search
        similar_categories = await self._find_similar_categories(text)
        
        # 2. LLM-based classification with context
        classification_result = await self._classify_with_llm(text, similar_categories)
        
        # 3. Validate and store results
        await self._validate_and_store_classification(state, classification_result, similar_categories)
        
        self.logger.info(f"Category classification complete: {classification_result.get('category', 'unknown')}")
        return state
    
    async def _find_similar_categories(self, text: str) -> List[Dict[str, Any]]:
        """Find similar categories using vector search"""
        try:
            # Check if Qdrant is properly initialized
            if not self.qdrant_client:
                self.logger.error("Qdrant client not initialized")
                return self._fallback_keyword_search(text)
            
            # Check if vector collection is initialized
            if not self._vector_collection_initialized:
                self.logger.warning("Vector collection not initialized, attempting to initialize...")
                await self._initialize_vector_collection()
                if not self._vector_collection_initialized:
                    return self._fallback_keyword_search(text)
            
            # Check collection exists
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
            except Exception:
                self.logger.warning("Collection not found, creating...")
                await self._initialize_vector_collection()
                if not self._vector_collection_initialized:
                    return self._fallback_keyword_search(text)
            
            # Get embedding for the input text
            query_embedding = await self._get_embedding(text)
            
            if not query_embedding:
                self.logger.warning("Failed to get embedding for query text")
                return self._fallback_keyword_search(text)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k_candidates * 2,  # Get more results to filter
                score_threshold=self.similarity_threshold
            )
            
            # Process results and group by main category
            category_scores = {}
            
            for result in search_results:
                category_name = result.payload["category_name"]
                score = result.score
                
                if category_name in category_scores:
                    # Take the best score for each category
                    category_scores[category_name] = max(category_scores[category_name], score)
                else:
                    category_scores[category_name] = score
            
            # Sort by score and return top candidates
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            
            similar_categories = []
            for category_name, score in sorted_categories[:self.top_k_candidates]:
                if self.hierarchy and category_name in self.hierarchy.categories:
                    category = self.hierarchy.categories[category_name]
                    similar_categories.append({
                        "name": category_name,
                        "description": category.description,
                        "similarity_score": score,
                        "subcategory_count": len(category.subcategories),
                        "keywords": list(category.keywords)
                    })
            
            self.logger.debug(f"Found {len(similar_categories)} similar categories")
            return similar_categories
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return self._fallback_keyword_search(text)
    
    def _fallback_keyword_search(self, text: str) -> List[Dict[str, Any]]:
        """Fallback keyword-based search when vector search fails"""
        if not self.hierarchy:
            return []
        
        # Simple keyword matching as fallback
        text_lower = text.lower()
        matches = []
        
        for category_name, category in self.hierarchy.categories.items():
            score = 0
            
            # Check category name
            if category_name.lower() in text_lower:
                score += 0.8
            
            # Check keywords
            for keyword in category.keywords:
                if keyword.lower() in text_lower:
                    score += 0.3
            
            if score > 0:
                matches.append({
                    "name": category_name,
                    "description": category.description,
                    "similarity_score": min(score, 1.0),
                    "subcategory_count": len(category.subcategories),
                    "keywords": list(category.keywords),
                    "search_method": "keyword_fallback"
                })
        
        # Sort by score and return top candidates
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:self.top_k_candidates]
    
    async def _classify_with_llm(self, text: str, similar_categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify using LLM with similar categories as context"""
        
        if not self.llm:
            raise ValueError("LLM not available for classification")
        
        # Build classification prompt
        prompt = self._build_classification_prompt(text, similar_categories)
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_message = SystemMessage(content="""
            أنت خبير في تصنيف تذاكر الدعم التقني لأنظمة المعلومات.
            قم بتصنيف النص المعطى إلى الفئة الرئيسية الأنسب.
            أجب بصيغة JSON فقط مع الحقول التالية:
            {"category": "اسم الفئة", "confidence": 0.95, "reasoning": "سبب الاختيار"}
            """)
            
            human_message = HumanMessage(content=prompt)
            
            response = await self._safe_llm_call([system_message, human_message])
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                
                # Validate required fields
                if "category" not in result:
                    raise ValueError("Missing category in response")
                
                # Ensure confidence is a float
                result["confidence"] = float(result.get("confidence", 0.5))
                
                # Add metadata
                result["classification_method"] = "llm_with_vector_context"
                result["similar_categories_used"] = len(similar_categories)
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
                
                # Fallback: extract category from text response
                return self._extract_category_from_text(response, similar_categories)
                
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            
            # Fallback to highest similarity category
            if similar_categories:
                return {
                    "category": similar_categories[0]["name"],
                    "confidence": similar_categories[0]["similarity_score"],
                    "reasoning": "Fallback to highest similarity match",
                    "classification_method": "vector_similarity_fallback"
                }
            else:
                return {
                    "category": "غير محدد",
                    "confidence": 0.1,
                    "reasoning": "No similar categories found",
                    "classification_method": "fallback_unknown"
                }
    
    def _build_classification_prompt(self, text: str, similar_categories: List[Dict[str, Any]]) -> str:
        """Build classification prompt with context"""
        
        prompt_parts = [
            "قم بتصنيف النص التالي إلى الفئة الرئيسية الأنسب:",
            f"النص: {text}",
            "",
            "الفئات المحتملة (مرتبة حسب التشابه):"
        ]
        
        for i, category in enumerate(similar_categories, 1):
            prompt_parts.append(
                f"{i}. {category['name']}: {category['description']} "
                f"(تشابه: {category['similarity_score']:.2f})"
            )
        
        prompt_parts.extend([
            "",
            "اختر الفئة الأنسب وأعط درجة الثقة (0-1) مع تبرير الاختيار.",
            "أجب بصيغة JSON فقط:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_category_from_text(self, response: str, similar_categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract category from text response as fallback"""
        
        # Look for category names in the response
        response_lower = response.lower()
        
        for category in similar_categories:
            if category["name"].lower() in response_lower:
                return {
                    "category": category["name"],
                    "confidence": 0.7,  # Medium confidence for text extraction
                    "reasoning": f"Extracted from response: {response[:100]}...",
                    "classification_method": "text_extraction_fallback"
                }
        
        # If no match found, use first similar category
        if similar_categories:
            return {
                "category": similar_categories[0]["name"],
                "confidence": 0.5,
                "reasoning": "Fallback to most similar category",
                "classification_method": "similarity_fallback"
            }
        
        return {
            "category": "غير محدد",
            "confidence": 0.1,
            "reasoning": "No categories found in response",
            "classification_method": "unknown_fallback"
        }
    
    async def _validate_and_store_classification(self, state: TicketState, 
                                               classification_result: Dict[str, Any],
                                               similar_categories: List[Dict[str, Any]]):
        """Validate classification result and store in state"""
        
        category = classification_result.get("category", "").strip()
        confidence = float(classification_result.get("confidence", 0.0))
        reasoning = classification_result.get("reasoning", "")
        
        # Validate category exists in hierarchy
        valid_category = None
        if self.hierarchy:
            # Exact match first
            if category in self.hierarchy.categories:
                valid_category = category
            else:
                # Try to find close match by name similarity
                category_lower = category.lower()
                for cat_name in self.hierarchy.categories.keys():
                    if cat_name.lower() == category_lower:
                        valid_category = cat_name
                        break
                    elif category_lower in cat_name.lower() or cat_name.lower() in category_lower:
                        valid_category = cat_name
                        confidence *= 0.9  # Slight confidence reduction for fuzzy match
                        break
        
        if not valid_category:
            self.logger.warning(f"Category '{category}' not found in hierarchy")
            
            # Fallback to most similar category
            if similar_categories:
                valid_category = similar_categories[0]["name"]
                confidence = min(confidence * 0.7, similar_categories[0]["similarity_score"])
                reasoning += f" (fallback from '{category}' to valid category)"
            else:
                valid_category = "عام"  # Default category
                confidence = 0.3
                reasoning = f"Invalid category '{category}', using default"
        
        # Store for compatibility with pipeline metrics
        state.category_confidence = confidence
        
        # Ensure classification object has proper structure
        if not hasattr(state, 'classification') or state.classification is None:
            from ..models.ticket_state import TicketClassification
            state.classification = TicketClassification()
        
        # Store in classification object (primary storage location)
        state.classification.main_category = valid_category
        state.classification.confidence_score = confidence
        
        # Add category description if available
        if self.hierarchy and valid_category in self.hierarchy.categories:
            state.classification.main_category_description = self.hierarchy.categories[valid_category].description
        
        # Store classification metadata
        if not hasattr(state, 'classification_metadata'):
            state.classification_metadata = {}
        
        state.classification_metadata['category_agent'] = {
            'processing_timestamp': datetime.now().isoformat(),
            'original_llm_category': category,
            'final_category': valid_category,
            'similar_categories_found': len(similar_categories),
            'classification_method': classification_result.get("classification_method"),
            'confidence_threshold_met': confidence >= self.config.confidence_threshold,
            'reasoning': reasoning
        }
        
        # Set overall confidence flag
        state.classification_metadata['requires_review'] = confidence < self.config.confidence_threshold
        
        self.logger.info(f"Stored classification: {valid_category} (confidence: {confidence:.2f})")
    
    def _validate_output_state(self, state: TicketState) -> None:
        """Validate category classification results"""
        super()._validate_output_state(state)
        
        # Category classification specific validations
        if not hasattr(state, 'classification') or not state.classification.main_category:
            raise ValueError("Main category classification not completed")
        
        if not isinstance(getattr(state.classification, 'confidence_score', 0), (int, float)):
            raise ValueError("Main confidence score not set properly")
    
    async def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics"""
        try:
            # Get Qdrant collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            stats = {
                'agent_metrics': self.metrics.dict(),
                'vector_collection': {
                    'vectors_count': collection_info.vectors_count,
                    'indexed_vectors_count': collection_info.indexed_vectors_count,
                    'points_count': collection_info.points_count
                },
                'classification_config': {
                    'embedding_model': self.embedding_model,
                    'top_k_candidates': self.top_k_candidates,
                    'similarity_threshold': self.similarity_threshold,
                    'confidence_threshold': self.config.confidence_threshold
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get classification stats: {e}")
            return {'error': str(e)}
    
    async def update_category_examples(self, category: str, positive_examples: List[str], 
                                     negative_examples: List[str] = None):
        """Update few-shot examples for a specific category"""
        try:
            # Store examples in cache
            if category not in self.few_shot_cache:
                self.few_shot_cache[category] = {"positive": [], "negative": []}
            
            self.few_shot_cache[category]["positive"].extend(positive_examples)
            
            if negative_examples:
                self.few_shot_cache[category]["negative"].extend(negative_examples)
            
            # Add examples to vector collection for improved similarity search
            points = []
            point_id = len(self.qdrant_client.scroll(self.collection_name, limit=1)[0])
            
            for example in positive_examples:
                embedding = await self._get_embedding(example)
                if embedding:
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "category_name": category,
                            "example_text": example,
                            "type": "positive_example",
                            "added_at": datetime.now().isoformat()
                        }
                    )
                    points.append(point)
                    point_id += 1
            
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.logger.info(f"Added {len(points)} examples for category '{category}'")
            
        except Exception as e:
            self.logger.error(f"Failed to update category examples: {e}")
