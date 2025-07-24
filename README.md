# ITSM Classification System

A multi-agent AI system for classifying Arabic IT service management tickets using LangChain and OpenAI.

## 🏗️ Architecture

This system uses a **Sequential Agent Architecture** where specialized AI agents process tickets through a pipeline:

1. **Orchestrator Agent** - Workflow management and coordination
2. **Arabic Processing Agent** - Language understanding and normalization  
3. **Category Classifier Agent** - Main category classification (Level 1)
4. **Subcategory Classifier Agent** - Subcategory classification (Level 2)
5. **Validation Agent** - Quality control and conflict resolution
6. **Learning Agent** - Continuous improvement and knowledge management

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies  
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

### 3. Test the Setup

```python
from src.config import create_default_agent_config
from src.agents import BaseAgent

# This will verify your setup is working
config = create_default_agent_config("test_agent")
print(f"✅ Configuration loaded: {config.agent_name}")
```

## 📁 Project Structure

```
itsm_classification/
├── .venv/                    # Virtual environment
├── src/
│   ├── __init__.py           # Package initialization
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py     # ✅ Base agent framework
│   ├── config/
│   │   ├── __init__.py  
│   │   └── agent_config.py   # ✅ Configuration system
│   └── models/               # 🔄 Coming in Step 2
├── .env                      # Environment variables (create from .env.example)
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 What You've Learned (Step 1)

### **Abstract Base Classes (ABC)**
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    async def process(self, state):
        pass  # Must be implemented by child classes
```
- **Interface Contract**: Guarantees all agents have a `process()` method
- **Early Error Detection**: Python errors at startup if method not implemented
- **Documentation**: Makes code self-documenting

### **Dataclasses (Modern Python)**
```python
@dataclass
class BaseAgentConfig:
    agent_name: str
    model_name: str = "gpt-4o"  # Default value
```
- **Less Boilerplate**: Auto-generates `__init__`, `__repr__`, `__eq__`
- **Type Hints**: Built-in support for type checking
- **Immutable Options**: Can use `frozen=True` for immutable configs

### **Async/Await (Modern Concurrency)**
```python
async def process(self, state):
    result = await self.llm.ainvoke(prompt)
    return result
```
- **Non-blocking**: Other operations run while waiting for LLM responses
- **Scalability**: Handle multiple tickets simultaneously
- **Future-proof**: Works with web frameworks like FastAPI

## 🔧 Current Implementation Status

- ✅ **Step 1**: Base Agent Infrastructure (COMPLETED)
  - BaseAgent abstract class with LLM integration
  - Configuration system with environment variable support
  - Metrics tracking and error handling
  - Project structure and virtual environment

- 🔄 **Step 2**: State Management System (NEXT)
  - TicketState schema for data flow between agents
  - State validation and persistence
  - Error recovery mechanisms

## 🎯 Next Steps

In Step 2, we'll implement:
1. **TicketState Schema** - Data structure that flows between agents
2. **State Management** - How agents pass data to each other
3. **Simple State Validation** - Ensure data integrity

## 🤝 Contributing

This is an educational project built step-by-step. Each step introduces new concepts and builds on previous work.

## 📚 Dependencies

- **openai**: Latest OpenAI Python client
- **langchain**: LLM application framework
- **langchain-openai**: OpenAI integration for LangChain  
- **langgraph**: Graph-based agent orchestration
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation and settings management
