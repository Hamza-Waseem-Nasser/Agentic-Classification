"""
STEP 3: ARABIC PROCESSING AGENT - LANGUAGE UNDERSTANDING & NORMALIZATION

This agent specializes in Arabic language processing, normalization, and entity
extraction for ITSM tickets. It prepares Arabic text for accurate classification
by handling dialect variations, technical terms, and linguistic complexities.

KEY RESPONSIBILITIES:
1. Arabic Text Normalization: Remove diacritics, standardize spelling variants
2. Named Entity Extraction: Identify technical terms, product names, error codes
3. Dialect Detection & Conversion: Handle Gulf, Levantine, Egyptian → MSA
4. Technical Term Preservation: Maintain ITSM-specific terminology intact
5. Language Quality Assessment: Evaluate text quality and processing confidence

ARABIC PROCESSING PIPELINE:
Input Text → Dialect Detection → Normalization → Entity Extraction → Technical Term ID → Quality Assessment

DESIGN DECISIONS:
- LLM-First Approach: Use GPT-4 for complex Arabic understanding
- Progressive Processing: Multiple stages for comprehensive text analysis
- Technical Term Protection: Preserve important ITSM terminology
- Confidence Scoring: Assess processing quality for downstream agents
- Fallback Strategies: Handle mixed languages and poor quality text

LINGUISTIC FEATURES:
- Diacritic Removal: Clean text while preserving meaning
- Spelling Standardization: Handle common Arabic spelling variations
- Entity Recognition: Extract names, places, technical terms, codes
- Dialect Normalization: Convert regional dialects to standard Arabic
- Mixed Language Handling: Process Arabic text with embedded English/numbers

INTEGRATION POINTS:
- CategoryLoader: Access technical glossary for term identification
- Vector Search: Prepare normalized text for semantic similarity
- Classification Agents: Provide clean, standardized text for accurate classification
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .base_agent import BaseAgent, BaseAgentConfig, AgentType
from ..models.ticket_state import TicketState


class ArabicProcessingAgent(BaseAgent):
    """
    Arabic Processing Agent: Specialized in Arabic language understanding.
    
    Handles the complexities of Arabic text processing including dialect
    normalization, entity extraction, and technical term identification.
    """
    
    def __init__(self, config: BaseAgentConfig, technical_glossary: Optional[Dict[str, str]] = None):
        super().__init__(config)
        self.technical_glossary = technical_glossary or self._load_default_glossary()
        self.arabic_patterns = self._initialize_arabic_patterns()
        
        # Initialize Arabic-specific configurations
        self.dialect_markers = self._load_dialect_markers()
        self.normalization_rules = self._load_normalization_rules()
        
        # Add system tag removal attributes (required for main.py integration)
        self.remove_system_tags = True
        self.system_tags_patterns = []  # Can be extended by pipeline
        
        self.logger.info("Arabic Processing Agent initialized with enhanced language support")
    
    def _remove_system_tags(self, text: str) -> str:
        """Remove system-generated tags and metadata"""
        # Remove common system tags
        system_patterns = [
            r'\(AutoClosed\)',
            r'\(auto[_\-]?closed\)',
            r'\[.*?closed.*?\]',
            r'تم الإغلاق تلقائي[اً]?',
            r'مغلق تلقائي[اً]?',
            r'\(.*?[Cc]losed.*?\)',
            r'\[.*?[Cc]losed.*?\]',
        ]
        
        cleaned = text
        for pattern in system_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace left by removals
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    def _load_default_glossary(self) -> Dict[str, str]:
        """Load default technical terms and their normalized forms"""
        return {
            # Common ITSM terms
            "سيستم": "نظام",
            "ابليكيشن": "تطبيق",
            "لوجين": "تسجيل الدخول",
            "باسوورد": "كلمة المرور",
            "ايميل": "بريد إلكتروني",
            "اكونت": "حساب",
            "سيرفر": "خادم",
            "داتابيس": "قاعدة البيانات",
            "نتورك": "شبكة",
            "فايروال": "جدار الحماية",
            
            # Status terms
            "اوفلاين": "غير متصل",
            "اونلاين": "متصل",
            "اكتيف": "نشط",
            "انكتيف": "غير نشط",
            
            # Action terms
            "ابديت": "تحديث",
            "انستول": "تثبيت",
            "داونلود": "تحميل",
            "ابلود": "رفع",
            "سيف": "حفظ",
            "ديليت": "حذف"
        }
    
    def _initialize_arabic_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for Arabic text processing"""
        return {
            # Arabic diacritics (Tashkeel)
            'diacritics': re.compile(r'[\u064B-\u065F\u0670\u0640]'),
            
            # Arabic letters range
            'arabic_letters': re.compile(r'[\u0600-\u06FF]+'),
            
            # Numbers and codes
            'numbers': re.compile(r'\d+'),
            'error_codes': re.compile(r'[A-Z]{2,}\d+|ERR\d+|\d{3,}'),
            
            # Email patterns
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # URL patterns
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            
            # Technical terms in mixed script
            'mixed_tech': re.compile(r'[A-Za-z]+[\u0600-\u06FF]+|[\u0600-\u06FF]+[A-Za-z]+')
        }
    
    def _load_dialect_markers(self) -> Dict[str, List[str]]:
        """Load dialect-specific markers for identification"""
        return {
            'gulf': [
                'شلون', 'وين', 'هاي', 'يالله', 'ماكو', 'أكو', 'گاع', 'ويا'
            ],
            'levantine': [
                'كيف', 'وين', 'هاد', 'هاي', 'منيح', 'مش', 'عم', 'بدي'
            ],
            'egyptian': [
                'ازاي', 'فين', 'ده', 'دي', 'مش', 'عايز', 'عاوز', 'هو'
            ],
            'maghrebi': [
                'كيفاش', 'فين', 'هادا', 'هادي', 'ماشي', 'بغيت', 'واش'
            ]
        }
    
    def _load_normalization_rules(self) -> List[Tuple[str, str]]:
        """Load text normalization rules"""
        return [
            # Alef variants
            ('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'),
            
            # Yaa variants
            ('ي', 'ي'), ('ى', 'ي'),
            
            # Taa marbouta
            ('ة', 'ه'),
            
            # Common spelling variations
            ('لايک', 'لايك'), ('سيستم', 'نظام'), ('ابليكشن', 'تطبيق'),
            
            # Remove extra spaces
            ('  +', ' '),
        ]
    
    async def process(self, state: TicketState) -> TicketState:
        """
        Process Arabic text through comprehensive language analysis pipeline.
        
        Args:
            state: Ticket state with original Arabic text
            
        Returns:
            Enhanced state with processed text and linguistic metadata
        """
        self.logger.info(f"Starting Arabic processing for ticket {state.ticket_id}")
        
        text = state.original_text
        
        # 1. Initial text quality assessment
        quality_metrics = await self._assess_text_quality(text)
        
        # 2. Dialect detection
        detected_dialect = await self._detect_dialect(text)
        
        # 3. Text normalization
        normalized_text = await self._normalize_arabic_text(text)
        
        # 4. Entity extraction
        entities = await self._extract_entities(normalized_text)
        
        # 5. Technical term identification
        technical_terms = await self._identify_technical_terms(normalized_text)
        
        # 6. Final processing confidence
        processing_confidence = await self._calculate_processing_confidence(
            quality_metrics, entities, technical_terms
        )
        
        # Update state with processing results
        state.processed_text = normalized_text
        state.extracted_keywords = [entity['text'] for entity in entities]
        state.language_confidence = processing_confidence
        state.entities = [entity['text'] for entity in entities]
        state.technical_terms = [term.get('term', term.get('original', '')) for term in technical_terms]
        
        # Add Arabic-specific metadata
        if not state.arabic_processing:
            state.arabic_processing = {}
        
        state.arabic_processing.update({
            'dialect_detected': detected_dialect,
            'quality_metrics': quality_metrics,
            'entities_found': len(entities),
            'technical_terms_found': len(technical_terms),
            'normalization_applied': True,
            'processing_confidence': processing_confidence,
            'extracted_entities': entities,
            'technical_terms': technical_terms
        })
        
        self.logger.info(f"Arabic processing complete: confidence={processing_confidence:.2f}, dialect={detected_dialect}")
        return state
    
    async def _assess_text_quality(self, text: str) -> Dict[str, float]:
        """Assess the quality of input text for processing"""
        
        quality_metrics = {
            'length_score': 0.0,
            'arabic_ratio': 0.0,
            'completeness_score': 0.0,
            'readability_score': 0.0
        }
        
        # Length assessment
        text_length = len(text.strip())
        if text_length >= 20:
            quality_metrics['length_score'] = min(text_length / 100, 1.0)
        else:
            quality_metrics['length_score'] = text_length / 20
        
        # Arabic content ratio
        arabic_chars = len(self.arabic_patterns['arabic_letters'].findall(text))
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0:
            quality_metrics['arabic_ratio'] = arabic_chars / total_chars
        
        # Completeness (presence of verbs, nouns indicators)
        completeness_indicators = ['لا', 'في', 'من', 'إلى', 'على', 'هذا', 'التي', 'الذي']
        found_indicators = sum(1 for indicator in completeness_indicators if indicator in text)
        quality_metrics['completeness_score'] = min(found_indicators / 3, 1.0)
        
        # Readability (absence of excessive repetition, presence of spaces)
        words = text.split()
        unique_words = set(words)
        if len(words) > 0:
            quality_metrics['readability_score'] = len(unique_words) / len(words)
        
        return quality_metrics
    
    async def _detect_dialect(self, text: str) -> str:
        """Detect Arabic dialect using marker words and LLM assistance"""
        
        text_lower = text.lower()
        dialect_scores = {}
        
        # Score based on dialect markers
        for dialect, markers in self.dialect_markers.items():
            score = sum(1 for marker in markers if marker in text_lower)
            if score > 0:
                dialect_scores[dialect] = score
        
        # If clear dialect markers found, return the highest scoring dialect
        if dialect_scores:
            detected_dialect = max(dialect_scores, key=dialect_scores.get)
            self.logger.debug(f"Dialect detected via markers: {detected_dialect}")
            return detected_dialect
        
        # Use LLM for more sophisticated dialect detection
        if self.llm and len(text.strip()) > 10:
            try:
                dialect_prompt = """
                حدد اللهجة العربية للنص التالي. أجب بكلمة واحدة فقط:
                - msa (العربية الفصحى)
                - gulf (خليجي)
                - levantine (شامي)
                - egyptian (مصري)
                - maghrebi (مغربي)
                
                النص: {text}
                
                اللهجة:""".format(text=text)
                
                from langchain_core.messages import HumanMessage
                response = await self._safe_llm_call([HumanMessage(content=dialect_prompt)])
                detected_dialect = response.content.strip().lower()
                
                if detected_dialect in ['msa', 'gulf', 'levantine', 'egyptian', 'maghrebi']:
                    self.logger.debug(f"Dialect detected via LLM: {detected_dialect}")
                    return detected_dialect
                    
            except Exception as e:
                self.logger.warning(f"LLM dialect detection failed: {e}")
        
        # Default to MSA if detection fails
        return 'msa'
    
    async def _normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text using rules and LLM assistance"""
        
        # FIRST: Remove system-generated tags
        text = self._remove_system_tags(text)
        
        normalized = text
        
        # 1. Remove diacritics
        normalized = self.arabic_patterns['diacritics'].sub('', normalized)
        
        # 2. Apply normalization rules
        for pattern, replacement in self.normalization_rules:
            normalized = re.sub(pattern, replacement, normalized)
        
        # 3. Normalize technical terms using glossary
        for term, normalized_form in self.technical_glossary.items():
            # Case-insensitive replacement
            normalized = re.sub(re.escape(term), normalized_form, normalized, flags=re.IGNORECASE)
        
        # 4. Clean up whitespace
        normalized = ' '.join(normalized.split())
        
        # 5. LLM-assisted normalization with STRICT instructions
        if self.llm and len(text) > 50:
            try:
                normalization_prompt = f"""
                قم بتنظيف وتوحيد النص التالي للمعالجة الآلية:
                
                القواعد المهمة:
                1. أزل العبارات العاطفية والشخصية
                2. حول إلى صيغة محايدة ومباشرة
                3. أزل التكرار والحشو
                4. احتفظ بالمعلومات التقنية المهمة فقط
                5. لا تضف أي معلومات جديدة
                6. اكتب النص النظيف فقط بدون شرح
                
                النص الأصلي: {normalized}
                
                النص النظيف:"""
                
                from langchain_core.messages import HumanMessage
                response = await self._safe_llm_call([HumanMessage(content=normalization_prompt)])
                llm_normalized = response.content.strip()
                
                # Use LLM result if it's reasonable
                if len(llm_normalized) > 0 and len(llm_normalized) < len(text) * 1.5:
                    normalized = llm_normalized
                    self.logger.debug("Applied LLM-assisted normalization")
                    
            except Exception as e:
                self.logger.warning(f"LLM normalization failed: {e}")
        
        return normalized
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from Arabic text"""
        
        entities = []
        
        # 1. Pattern-based extraction
        
        # Extract emails
        emails = self.arabic_patterns['email'].findall(text)
        for email in emails:
            entities.append({
                'text': email,
                'type': 'email',
                'confidence': 0.95,
                'start_pos': text.find(email)
            })
        
        # Extract URLs
        urls = self.arabic_patterns['url'].findall(text)
        for url in urls:
            entities.append({
                'text': url,
                'type': 'url',
                'confidence': 0.95,
                'start_pos': text.find(url)
            })
        
        # Extract error codes
        error_codes = self.arabic_patterns['error_codes'].findall(text)
        for code in error_codes:
            entities.append({
                'text': code,
                'type': 'error_code',
                'confidence': 0.90,
                'start_pos': text.find(code)
            })
        
        # Extract numbers (potential IDs, phone numbers, etc.)
        numbers = self.arabic_patterns['numbers'].findall(text)
        for number in numbers:
            if len(number) >= 3:  # Ignore single digits
                entities.append({
                    'text': number,
                    'type': 'number',
                    'confidence': 0.80,
                    'start_pos': text.find(number)
                })
        
        # 2. LLM-based entity extraction with better JSON handling
        if self.llm and len(text) > 20:
            try:
                entity_prompt = f"""استخرج الكيانات المهمة من النص التالي.

النص: {text}

أنواع الكيانات المطلوبة:
- أسماء الأشخاص
- أسماء الشركات  
- أسماء المنتجات أو الأنظمة
- أرقام مرجعية
- مصطلحات تقنية

الإجابة يجب أن تكون JSON array فقط، مثال:
[{{"text": "كلمة", "type": "نوع", "confidence": 0.9}}]

إذا لم تجد كيانات، أرجع: []

JSON:"""
                
                from langchain_core.messages import HumanMessage
                response = await self._safe_llm_call([HumanMessage(content=entity_prompt)])
                
                # Clean response to ensure valid JSON
                json_str = response.content.strip()
                
                # Try to extract JSON array from response
                if '[' in json_str and ']' in json_str:
                    start = json_str.find('[')
                    end = json_str.rfind(']') + 1
                    json_str = json_str[start:end]
                    
                    try:
                        llm_entities = json.loads(json_str)
                        if isinstance(llm_entities, list):
                            for entity in llm_entities:
                                if isinstance(entity, dict) and 'text' in entity and 'type' in entity:
                                    entity['start_pos'] = text.find(entity['text'])
                                    entities.append(entity)
                            self.logger.debug(f"LLM extracted {len(llm_entities)} entities")
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse cleaned JSON response")
                else:
                    self.logger.warning("No JSON array found in LLM response")
                    
            except Exception as e:
                self.logger.warning(f"LLM entity extraction failed: {e}")
        
        # Remove duplicates and sort by position
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            if entity['text'] not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(entity['text'])
        
        unique_entities.sort(key=lambda x: x['start_pos'])
        
        return unique_entities
    
    async def _identify_technical_terms(self, text: str) -> List[Dict[str, Any]]:
        """Identify technical terms and ITSM-specific vocabulary"""
        
        technical_terms = []
        text_lower = text.lower()
        
        # 1. Check against known technical glossary
        for term, normalized_form in self.technical_glossary.items():
            if term in text_lower:
                technical_terms.append({
                    'term': term,
                    'normalized': normalized_form,
                    'category': 'known_technical',
                    'confidence': 0.95
                })
        
        # 2. Pattern-based technical term detection
        technical_patterns = [
            r'[A-Za-z]+\.exe',  # Executable files
            r'[A-Za-z]+\.dll',  # DLL files
            r'[A-Za-z]+\.com',  # COM domains
            r'[A-Za-z]+\.org',  # ORG domains
            r'HTTP[S]?',        # HTTP protocols
            r'FTP[S]?',         # FTP protocols
            r'SQL',             # Database terms
            r'API',             # API terms
        ]
        
        for pattern in technical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                technical_terms.append({
                    'term': match.group(),
                    'normalized': match.group().upper(),
                    'category': 'pattern_detected',
                    'confidence': 0.85
                })
        
        # 3. LLM-based technical term identification with better JSON handling
        if self.llm and len(text) > 30:
            try:
                tech_prompt = f"""حدد المصطلحات التقنية في النص التالي.

النص: {text}

ابحث عن:
- مصطلحات البرمجة والحاسوب
- أسماء الأنظمة والتطبيقات
- مصطلحات الشبكات وقواعد البيانات

أجب بـ JSON array فقط:
[{{"term": "المصطلح", "category": "الفئة", "confidence": 0.9}}]

إذا لم تجد مصطلحات، أرجع: []

JSON:"""
                
                from langchain_core.messages import HumanMessage
                response = await self._safe_llm_call([HumanMessage(content=tech_prompt)])
                
                # Extract and clean JSON
                json_str = response.content.strip()
                if '[' in json_str and ']' in json_str:
                    start = json_str.find('[')
                    end = json_str.rfind(']') + 1
                    json_str = json_str[start:end]
                    
                    try:
                        llm_terms = json.loads(json_str)
                        if isinstance(llm_terms, list):
                            for term_data in llm_terms:
                                if isinstance(term_data, dict) and 'term' in term_data:
                                    technical_terms.append({
                                        'term': term_data['term'],
                                        'category': term_data.get('category', 'llm_detected'),
                                        'confidence': term_data.get('confidence', 0.80)
                                    })
                            self.logger.debug(f"LLM identified {len(llm_terms)} technical terms")
                    except json.JSONDecodeError:
                        self.logger.warning("Failed to parse technical terms JSON")
                        
            except Exception as e:
                self.logger.warning(f"LLM technical term identification failed: {e}")
        
        # Remove duplicates
        unique_terms = []
        seen_terms = set()
        
        for term in technical_terms:
            term_key = term['term'].lower()
            if term_key not in seen_terms:
                unique_terms.append(term)
                seen_terms.add(term_key)
        
        return unique_terms
    
    async def _calculate_processing_confidence(self, quality_metrics: Dict[str, float], 
                                             entities: List[Dict], 
                                             technical_terms: List[Dict]) -> float:
        """Calculate overall processing confidence score"""
        
        # Base confidence from text quality
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        # Boost confidence if entities were found
        entity_boost = min(len(entities) * 0.1, 0.3)
        
        # Boost confidence if technical terms were found
        tech_boost = min(len(technical_terms) * 0.05, 0.2)
        
        # Arabic content bonus
        arabic_bonus = quality_metrics.get('arabic_ratio', 0) * 0.2
        
        # Calculate final confidence
        confidence = quality_score + entity_boost + tech_boost + arabic_bonus
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _validate_output_state(self, state: TicketState) -> None:
        """Validate Arabic processing results"""
        super()._validate_output_state(state)
        
        # Arabic processing specific validations
        if not state.processed_text:
            raise ValueError("Processed text not generated")
        
        if state.language_confidence <= 0:
            raise ValueError("Language confidence not calculated")
        
        if not hasattr(state, 'arabic_processing'):
            raise ValueError("Arabic processing metadata not set")
