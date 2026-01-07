
import sys
import os

# Add parent directory to path to allow importing poly16z as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from poly16z.agent.openai_agent import OpenAIAgent
from poly16z.agent.gemini_agent import GeminiAgent
from poly16z.api.client import PolymarketClient
from poly16z.risk.manager import RiskManager

def test_agent_loading():
    print("Verifying Agent Instantiation...")
    
    client = PolymarketClient() # No keys
    risk = RiskManager()
    
    # Test OpenAI
    try:
        agent_o1 = OpenAIAgent(client, risk, "mock_key", "mock_prompt", model="o1-preview")
        print("✅ OpenAIAgent initialized (o1-preview mode)")
        
        agent_gpt = OpenAIAgent(client, risk, "mock_key", "mock_prompt", model="gpt-4o")
        print("✅ OpenAIAgent initialized (gpt-4o mode)")
    except Exception as e:
        print(f"❌ OpenAIAgent failed: {e}")
        
    # Test Gemini
    try:
        agent_gem = GeminiAgent(client, risk, "mock_key", "mock_prompt", model="gemini-1.5-pro")
        print("✅ GeminiAgent initialized (1.5-pro mode)")
    except Exception as e:
        print(f"❌ GeminiAgent failed: {e}")
        
    print("Agent Verification Complete")

if __name__ == "__main__":
    test_agent_loading()
