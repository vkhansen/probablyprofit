"""
Gemini Agent

AI-powered trading agent using Google's Gemini models.
"""

import json
from typing import Optional
import google.generativeai as genai
from loguru import logger

from poly16z.agent.base import BaseAgent, Observation, Decision
from poly16z.api.client import PolymarketClient
from poly16z.risk.manager import RiskManager


class GeminiAgent(BaseAgent):
    """
    AI-powered trading agent using Google's Gemini models.
    """

    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: RiskManager,
        google_api_key: str,
        strategy_prompt: str,
        model: str = "gemini-2.0-flash-exp",
        name: str = "GeminiAgent",
        loop_interval: int = 60,
    ):
        super().__init__(client, risk_manager, name, loop_interval)

        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={"response_mime_type": "application/json"}
        )
        self.strategy_prompt = strategy_prompt
        
        logger.info(f"GeminiAgent '{name}' initialized with model {model}")

    def _format_observation(self, observation: Observation) -> str:
        # Concise formatting for Gemini
        return f"""
        Timestamp: {observation.timestamp}
        Balance: ${observation.balance}
        Markets: {len(observation.markets)} available
        Positions: {len(observation.positions)} open
        
        Recent Activity:
        {self.memory.get_recent_history(3)}
        """

    async def decide(self, observation: Observation) -> Decision:
        logger.info(f"[{self.name}] Asking Gemini for decision...")

        try:
            obs_text = self._format_observation(observation)
            
            prompt = f"""
            {self.strategy_prompt}
            
            CURRENT SITUATION:
            {obs_text}
            
            Respond with a JSON object with this schema:
            {{
                "action": "buy" | "sell" | "hold",
                "market_id": "string",
                "outcome": "string",
                "size": float,
                "price": float,
                "reasoning": "string",
                "confidence": float
            }}
            """

            # Note: The generation_config in init enforces JSON mode
            response = self.model.generate_content(prompt)
            
            content = response.text
            logger.debug(f"Gemini response: {content[:200]}...")

            data = json.loads(content)
            
            return Decision(
                action=data.get("action", "hold"),
                market_id=data.get("market_id"),
                outcome=data.get("outcome"),
                size=float(data.get("size", 0)),
                price=float(data.get("price", 0)) if data.get("price") else None,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5))
            )

        except Exception as e:
            logger.error(f"Error getting decision from Gemini: {e}")
            return Decision(action="hold", reasoning=f"Error: {e}")
