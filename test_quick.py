"""
Quick test of the pipeline with just one ticket
"""
import asyncio
from src.models.ticket_state import TicketState
from src.agents.orchestrator_agent import OrchestratorAgent
from src.config.agent_config import BaseAgentConfig

async def test_orchestrator():
    """Test just the orchestrator agent"""
    config = BaseAgentConfig(agent_name="orchestrator")
    
    # Create ticket state
    state = TicketState(
        original_text="الكمبيوتر لا يعمل والشاشة سوداء، أحتاج مساعدة عاجلة"
    )
    
    print(f"Original state: {state.ticket_id}")
    
    # Test orchestrator
    try:
        orchestrator = OrchestratorAgent(config)
        result = await orchestrator.process(state)
        print(f"Orchestrator processing successful")
        print(f"Processing metadata: {result.processing_metadata}")
        print(f"Routing decisions: {result.routing_decisions}")
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
