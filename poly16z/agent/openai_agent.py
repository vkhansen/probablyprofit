"""
OpenAI Agent

AI-powered trading agent using GPT-4 for decision-making.
"""

import json
from typing import Optional
from openai import OpenAI
from loguru import logger

from poly16z.agent.base import BaseAgent, Observation, Decision
from poly16z.api.client import PolymarketClient
from poly16z.risk.manager import RiskManager


class OpenAIAgent(BaseAgent):
    """
    AI-powered trading agent using OpenAI's GPT models.
    """

    def __init__(
        self,
        client: PolymarketClient,
        risk_manager: RiskManager,
        openai_api_key: str,
        strategy_prompt: str,
        model: str = "gpt-4o",
        name: str = "OpenAIAgent",
        loop_interval: int = 60,
        temperature: float = 0.7,
    ):
        super().__init__(client, risk_manager, name, loop_interval)

        self.openai = OpenAI(api_key=openai_api_key)
        self.strategy_prompt = strategy_prompt
        self.model = model
        self.temperature = temperature

        logger.info(f"OpenAIAgent '{name}' initialized with model {model}")

    def _format_observation(self, observation: Observation) -> str:
        # Reuse similar formatting logic or create a shared utility in future refactor
        # For now, copying to keep agents independent
        
        markets_info = []
        for market in observation.markets[:20]:
            markets_info.append(
                f"Market: {market.question}\n"
                f"  ID: {market.condition_id}\n"
                f"  Outcomes: {', '.join(market.outcomes)}\n"
                f"  Prices: {', '.join(f'{p:.2%}' for p in market.outcome_prices)}\n"
                f"  Volume: ${market.volume:,.0f}\n"
                f"  Liquidity: ${market.liquidity:,.0f}\n"
            )

        positions_info = []
        for pos in observation.positions:
            positions_info.append(
                f"Position in {pos.market_id}:\n"
                f"  Outcome: {pos.outcome}\n"
                f"  Size: {pos.size:.2f}\n"
                f"  P&L: ${pos.unrealized_pnl:.2f}\n"
            )

        prompt = f"""Current Market State:
Time: {observation.timestamp}
Balance: ${observation.balance:,.2f}

Positions:
{chr(10).join(positions_info) if positions_info else "None"}

Top Markets:
{chr(10).join(markets_info)}

History:
{self.memory.get_recent_history(5)}
"""
        return prompt

    async def decide(self, observation: Observation) -> Decision:
        logger.info(f"[{self.name}] Asking GPT-4 for decision...")

        try:
            obs_text = self._format_observation(observation)
            
            system_prompt = f"""{self.strategy_prompt}

You are a trading bot. Respond in strict JSON format.
Output schema:
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

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            logger.debug(f"GPT response: {content[:200]}...")

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
            logger.error(f"Error getting decision from OpenAI: {e}")
            return Decision(action="hold", reasoning=f"Error: {e}")
