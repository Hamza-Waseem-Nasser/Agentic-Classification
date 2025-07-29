"""
🧠 PRACTICAL MEMORY AND CONTEXT DEMONSTRATION
Educational Code Examples for Multi-Agent ITSM System

This file shows exactly how memory and context flow through your agents
with real code examples and detailed explanations.
"""

from typing import Dict, Any, List
from datetime import datetime
import json

# ==============================================================================
# 1. SHARED STATE MEMORY EXAMPLE
# ==============================================================================

class TicketStateDemo:
    """
    Demonstrates how shared state flows through agents.
    This is the PRIMARY memory sharing mechanism in your system.
    """
    
    def __init__(self):
        # Initial state when ticket enters pipeline
        self.initial_state = {
            "ticket_id": "demo_ticket_001",
            "original_text": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله",
            "created_at": datetime.now().isoformat(),
            
            # Empty fields to be filled by agents
            "processed_text": None,
            "classification": {},
            "metadata": {},
            "processing_metadata": {},
            "arabic_processing": {},
            "classification_metadata": {}
        }
    
    def demonstrate_state_evolution(self):
        """Shows how state evolves through each agent"""
        
        print("🎯 INITIAL STATE:")
        print(json.dumps(self.initial_state, indent=2, ensure_ascii=False))
        
        # AGENT 1: Orchestrator adds routing and business logic
        after_orchestrator = self.initial_state.copy()
        after_orchestrator.update({
            "priority": "normal",
            "routing_decisions": {
                "complexity_score": 0.35,
                "processing_path": "standard",
                "priority_detected": False,
                "complexity_factors": {
                    "text_length": 59,
                    "has_technical_terms": True,
                    "has_arabic_mixed_languages": False
                }
            },
            "processing_metadata": {
                "orchestrator_start": datetime.now().isoformat(),
                "hierarchy_loaded": True,
                "total_categories": 19,
                "timeouts": {"arabic_processing": 30, "category_classification": 45}
            }
        })
        
        print("\n🎭 AFTER ORCHESTRATOR:")
        print(json.dumps(after_orchestrator, indent=2, ensure_ascii=False))
        
        # AGENT 2: Arabic Processor adds linguistic analysis
        after_arabic = after_orchestrator.copy()
        after_arabic.update({
            "processed_text": "تم تسجيل الدخول واستكمال البيانات، وتبين أن الشركة مسجلة.",
            "technical_terms": ["تسجيل الدخول", "البيانات"],
            "language_confidence": 0.59,
            "entities": [],
            "arabic_processing": {
                "dialect_detected": "msa",
                "normalization_applied": True,
                "technical_terms_found": 2,
                "processing_confidence": 0.59,
                "quality_metrics": {
                    "arabic_ratio": 0.95,
                    "length_score": 0.7
                }
            }
        })
        
        print("\n🔤 AFTER ARABIC PROCESSOR:")
        print(json.dumps(after_arabic, indent=2, ensure_ascii=False))
        
        # AGENT 3: Category Classifier adds main classification
        after_category = after_arabic.copy()
        after_category.update({
            "classification": {
                "main_category": "التسجيل",
                "main_category_description": "مشاكل التسجيل في النظام",
                "confidence_score": 0.85
            },
            "category_confidence": 0.85,  # Backward compatibility
            "classification_metadata": {
                "category_agent": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "similar_categories_found": 3,
                    "classification_method": "llm_with_vector_context",
                    "reasoning": "النص يتحدث عن مشكلة في عملية التسجيل"
                }
            }
        })
        
        print("\n🏷️ AFTER CATEGORY CLASSIFIER:")
        print(json.dumps(after_category, indent=2, ensure_ascii=False))
        
        # AGENT 4: Subcategory Classifier completes classification
        final_state = after_category.copy()
        final_state.update({
            "classification": {
                **after_category["classification"],
                "subcategory": "التحقق من السجل التجاري",
                "subcategory_description": "مشاكل في التحقق من السجل التجاري",
                "confidence_score": 0.8  # Final overall confidence
            },
            "subcategory_confidence": 0.8,
            "processing_completed": datetime.now().isoformat(),
            "classification_metadata": {
                **after_category["classification_metadata"],
                "subcategory_agent": {
                    "parent_category": "التسجيل",
                    "subcategories_considered": 3,
                    "reasoning": "النص يشير إلى مشكلة في التحقق من السجل التجاري"
                }
            }
        })
        
        print("\n🔍 FINAL STATE (After Subcategory Classifier):")
        print(json.dumps(final_state, indent=2, ensure_ascii=False))


# ==============================================================================
# 2. INDIVIDUAL AGENT MEMORY EXAMPLES
# ==============================================================================

class AgentMemoryDemo:
    """
    Demonstrates individual agent memory - private to each agent.
    """
    
    def __init__(self):
        # Each agent has its own private memory
        self.orchestrator_memory = {
            "business_rules": {
                "priority_keywords": ["عاجل", "طارئ", "مهم", "سريع"],
                "confidence_thresholds": {
                    "high_confidence": 0.85,
                    "medium_confidence": 0.65,
                    "low_confidence": 0.45
                }
            },
            "category_loader": "Reference to CategoryLoader instance",
            "state_manager": "Reference to StateManager instance"
        }
        
        self.arabic_processor_memory = {
            "technical_glossary": {
                "تسجيل الدخول": "login", 
                "البيانات": "data",
                "النظام": "system",
                "التطبيق": "application",
                "كلمة المرور": "password",
                "حساب": "account"
            },
            "dialect_markers": {
                "gulf": ["شلون", "وش", "هذا", "ذاك"],
                "levantine": ["كيف", "وين", "هاد", "هاي"],
                "egyptian": ["ازاي", "فين", "ده", "دي"]
            },
            "normalization_rules": [
                ("أ", "ا"), ("إ", "ا"), ("آ", "ا"),  # Alef variants
                ("ى", "ي"),  # Yaa variants
                ("ة", "ه")   # Taa marbouta
            ]
        }
        
        self.category_classifier_memory = {
            "qdrant_client": "QdrantClient instance",
            "collection_name": "itsm_categories",
            "embedding_model": "text-embedding-3-small",
            "top_k_candidates": 5,
            "similarity_threshold": 0.5,
            "few_shot_cache": {
                "التسجيل": {
                    "positive_examples": [
                        "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله",
                        "لا أستطيع إكمال عملية التسجيل"
                    ]
                }
            }
        }
        
        self.subcategory_classifier_memory = {
            "hierarchy": "Reference to ClassificationHierarchy",
            "top_k_subcategories": 3,
            "minimum_confidence": 0.6,
            "category_configs": {
                "التسجيل": {
                    "subcategories": ["التحقق من السجل التجاري", "مشاكل البيانات"],
                    "default_confidence": 0.7
                }
            }
        }
    
    def demonstrate_memory_isolation(self):
        """Shows how each agent has private memory"""
        
        print("🎭 ORCHESTRATOR PRIVATE MEMORY:")
        print(json.dumps(self.orchestrator_memory, indent=2, ensure_ascii=False))
        
        print("\n🔤 ARABIC PROCESSOR PRIVATE MEMORY:")
        print(json.dumps(self.arabic_processor_memory, indent=2, ensure_ascii=False))
        
        print("\n🏷️ CATEGORY CLASSIFIER PRIVATE MEMORY:")
        print(json.dumps(self.category_classifier_memory, indent=2, ensure_ascii=False))
        
        print("\n🔍 SUBCATEGORY CLASSIFIER PRIVATE MEMORY:")
        print(json.dumps(self.subcategory_classifier_memory, indent=2, ensure_ascii=False))


# ==============================================================================
# 3. VECTOR SEMANTIC MEMORY EXAMPLE
# ==============================================================================

class VectorMemoryDemo:
    """
    Demonstrates vector embeddings stored in Qdrant database.
    This is the SEMANTIC MEMORY that enables intelligent similarity search.
    """
    
    def __init__(self):
        # What gets embedded and stored in Qdrant
        self.embedded_content = {
            "categories": [
                {
                    "id": 1,
                    "category_name": "التسجيل",
                    "text_to_embed": "التسجيل مشاكل التسجيل في النظام وإنشاء حسابات جديدة",
                    "embedding": "[1536 float values]",  # Actual embedding from OpenAI
                    "metadata": {
                        "type": "main_category",
                        "subcategory_count": 5,
                        "keywords": ["تسجيل", "حساب", "إنشاء"]
                    }
                },
                {
                    "id": 2,
                    "category_name": "تسجيل الدخول", 
                    "text_to_embed": "تسجيل الدخول مشاكل الدخول للمنصة وكلمة المرور",
                    "embedding": "[1536 float values]",
                    "metadata": {
                        "type": "main_category",
                        "subcategory_count": 3,
                        "keywords": ["دخول", "لوجين", "كلمة مرور"]
                    }
                }
            ],
            "subcategories": [
                {
                    "id": 101,
                    "category_name": "التسجيل",
                    "subcategory_name": "التحقق من السجل التجاري",
                    "text_to_embed": "التحقق من السجل التجاري مشاكل في التحقق من السجل التجاري",
                    "embedding": "[1536 float values]",
                    "metadata": {
                        "type": "subcategory_context",
                        "parent_category": "التسجيل"
                    }
                }
            ],
            "training_examples": [
                {
                    "id": 1001,
                    "category_name": "التسجيل",
                    "example_text": "بعد تسجيل دخول واستكمال البيانات ظهر لي ان الشركه مسجله",
                    "embedding": "[1536 float values]",
                    "metadata": {
                        "type": "training_example",
                        "category": "التسجيل"
                    }
                }
            ]
        }
    
    def demonstrate_vector_search_process(self):
        """Shows how vector search works step by step"""
        
        print("🧠 VECTOR SEARCH PROCESS:")
        print("=" * 50)
        
        # Step 1: Query comes in
        query_text = "تم تسجيل الدخول واستكمال البيانات، وتبين أن الشركة مسجلة."
        print(f"1. Query Text: {query_text}")
        
        # Step 2: Generate embedding for query
        print("2. Generate Embedding:")
        print("   - API Call: POST https://api.openai.com/v1/embeddings")
        print("   - Model: text-embedding-3-small")
        print("   - Result: [1536 dimensional vector]")
        
        # Step 3: Search in Qdrant
        print("3. Vector Search in Qdrant:")
        print("   - Collection: itsm_categories")
        print("   - Query Vector: [1536 dimensions]")
        print("   - Search Method: Cosine Similarity")
        print("   - Limit: 10 results")
        print("   - Threshold: 0.5")
        
        # Step 4: Results
        search_results = [
            {"category": "التسجيل", "similarity_score": 0.87, "metadata": {"type": "main_category"}},
            {"category": "التحقق من السجل التجاري", "similarity_score": 0.82, "metadata": {"type": "subcategory"}},
            {"category": "تسجيل الدخول", "similarity_score": 0.65, "metadata": {"type": "main_category"}},
            {"category": "بيانات المنشأة", "similarity_score": 0.45, "metadata": {"type": "main_category"}}
        ]
        
        print("4. Search Results:")
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result['category']} (score: {result['similarity_score']})")
        
        # Step 5: Context for LLM
        print("5. Context Provided to LLM:")
        top_categories = [r['category'] for r in search_results if r['metadata']['type'] == 'main_category'][:3]
        print(f"   - Top Similar Categories: {top_categories}")
        print("   - This context helps LLM make informed decision")


# ==============================================================================
# 4. LLM CONVERSATION MEMORY EXAMPLE  
# ==============================================================================

class LLMContextDemo:
    """
    Demonstrates how LLM conversation memory works within each agent.
    """
    
    def __init__(self):
        # Example conversation memory for Category Classifier
        self.category_classifier_conversation = [
            {
                "role": "system",
                "content": """
                أنت خبير في تصنيف تذاكر الدعم التقني لأنظمة المعلومات.
                
                مهم جداً: يجب عليك اختيار فئة من القائمة المحددة فقط.
                
                الفئات المتاحة فقط هي:
                التسجيل, تسجيل الدخول, بيانات المنشأة, إضافة المنتجات
                
                استخدم Chain-of-Thought reasoning:
                1. أولاً، اقرأ النص وحدد الكلمات المفتاحية
                2. ثانياً، قارن مع الفئات المتاحة  
                3. ثالثاً، اختر الفئة الأنسب بناءً على التحليل
                """
            },
            {
                "role": "user",
                "content": """
                النص: تم تسجيل الدخول واستكمال البيانات، وتبين أن الشركة مسجلة.
                
                الفئات المشابهة من البحث الدلالي:
                1. التسجيل (similarity: 0.87)
                2. تسجيل الدخول (similarity: 0.65)
                3. بيانات المنشأة (similarity: 0.45)
                
                صنف النص واختر الفئة الصحيحة.
                """
            },
            {
                "role": "assistant", 
                "content": """
                {
                    "category": "التسجيل",
                    "confidence": 0.85,
                    "reasoning": "النص يتحدث عن مشكلة في عملية التسجيل حيث تم اكتشاف أن الشركة مسجلة مسبقاً",
                    "chain_of_thought": "أولاً: الكلمات المفتاحية هي 'تسجيل' و'البيانات' و'الشركة مسجلة'. ثانياً: مقارنة مع الفئات المتاحة تظهر أن 'التسجيل' هو الأنسب لأن المشكلة تتعلق بعملية التسجيل وليس تسجيل الدخول. ثالثاً: الثقة عالية 0.85 لأن السياق واضح والكلمات المفتاحية تؤكد هذا التصنيف."
                }
                """
            }
        ]
    
    def demonstrate_llm_context_evolution(self):
        """Shows how LLM context builds up during conversation"""
        
        print("🤖 LLM CONVERSATION MEMORY EVOLUTION:")
        print("=" * 60)
        
        for i, message in enumerate(self.category_classifier_conversation, 1):
            print(f"\nMessage {i} ({message['role'].upper()}):")
            print("-" * 30)
            print(message['content'][:200] + "..." if len(message['content']) > 200 else message['content'])
        
        print("\n💡 KEY INSIGHTS:")
        print("- System message sets the expert role and constraints")
        print("- User message provides text + vector search context")  
        print("- Assistant response includes reasoning and chain-of-thought")
        print("- Each API call is independent (no conversation persistence)")


# ==============================================================================
# 5. CONTEXT SHARING PATTERNS DEMONSTRATION
# ==============================================================================

class ContextSharingDemo:
    """
    Demonstrates the different patterns of context sharing between agents.
    """
    
    def demonstrate_hierarchical_context(self):
        """Shows how subcategory agent inherits context from category agent"""
        
        print("🏗️ HIERARCHICAL CONTEXT INHERITANCE:")
        print("=" * 50)
        
        # Context from previous agents available to subcategory classifier
        inherited_context = {
            "from_orchestrator": {
                "priority": "normal",
                "complexity_score": 0.35,
                "processing_path": "standard"
            },
            "from_arabic_processor": {
                "processed_text": "تم تسجيل الدخول واستكمال البيانات، وتبين أن الشركة مسجلة.",
                "technical_terms": ["تسجيل الدخول", "البيانات"],
                "language_confidence": 0.59,
                "dialect_detected": "msa"
            },
            "from_category_classifier": {
                "main_category": "التسجيل",
                "category_confidence": 0.85,
                "similar_categories": ["تسجيل الدخول", "بيانات المنشأة"],
                "classification_reasoning": "النص يتحدث عن مشكلة في عملية التسجيل"
            }
        }
        
        print("Context Available to Subcategory Classifier:")
        print(json.dumps(inherited_context, indent=2, ensure_ascii=False))
        
        # How subcategory agent uses this context
        subcategory_decision_process = {
            "step_1_filter_subcategories": {
                "parent_category": "التسجيل",
                "available_subcategories": [
                    "التحقق من السجل التجاري",
                    "مشاكل البيانات", 
                    "إنشاء حساب جديد"
                ]
            },
            "step_2_analyze_context": {
                "key_phrases": ["الشركة مسجلة", "استكمال البيانات"],
                "technical_terms": ["تسجيل الدخول", "البيانات"],
                "category_confidence": 0.85  # High confidence from previous agent
            },
            "step_3_make_decision": {
                "chosen_subcategory": "التحقق من السجل التجاري",
                "reasoning": "العبارة 'الشركة مسجلة' تشير إلى مشكلة في التحقق من السجل التجاري"
            }
        }
        
        print("\nSubcategory Decision Process:")
        print(json.dumps(subcategory_decision_process, indent=2, ensure_ascii=False))


# ==============================================================================
# 6. COMPREHENSIVE DEMO RUNNER
# ==============================================================================

def run_memory_and_context_demo():
    """
    Runs all demonstrations to show complete memory and context flow.
    """
    
    print("🧠 MULTI-AGENT MEMORY AND CONTEXT DEMONSTRATION")
    print("=" * 80)
    print("This demo shows exactly how memory and context flow through your agents.")
    print("=" * 80)
    
    # Demo 1: Shared State Evolution
    print("\n\n1️⃣ SHARED STATE MEMORY EVOLUTION")
    print("-" * 40)
    state_demo = TicketStateDemo()
    state_demo.demonstrate_state_evolution()
    
    # Demo 2: Individual Agent Memory
    print("\n\n2️⃣ INDIVIDUAL AGENT MEMORY")
    print("-" * 40)
    agent_demo = AgentMemoryDemo()
    agent_demo.demonstrate_memory_isolation()
    
    # Demo 3: Vector Semantic Memory
    print("\n\n3️⃣ VECTOR SEMANTIC MEMORY")
    print("-" * 40)
    vector_demo = VectorMemoryDemo()
    vector_demo.demonstrate_vector_search_process()
    
    # Demo 4: LLM Conversation Memory
    print("\n\n4️⃣ LLM CONVERSATION MEMORY")
    print("-" * 40)
    llm_demo = LLMContextDemo()
    llm_demo.demonstrate_llm_context_evolution()
    
    # Demo 5: Context Sharing Patterns
    print("\n\n5️⃣ CONTEXT SHARING PATTERNS")
    print("-" * 40)
    context_demo = ContextSharingDemo()
    context_demo.demonstrate_hierarchical_context()
    
    print("\n\n✅ DEMO COMPLETE!")
    print("You now understand how memory and context flow through your multi-agent system.")


if __name__ == "__main__":
    run_memory_and_context_demo()
