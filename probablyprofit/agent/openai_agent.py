"""
OpenAI Agent

AI-powered trading agent using GPT-4 for decision-making.
"""

import asyncio
from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from probablyprofit.agent.base import BaseAgent, Decision, Observation
from probablyprofit.agent.formatters import ObservationFormatter, get_decision_schema
from probablyprofit.api.client import PolymarketClient
from probablyprofit.api.exceptions import (
    AgentException,
    NetworkException,
    SchemaValidationError,
)
from probablyprofit.risk.manager import RiskManager
from probablyprofit.utils.validation_utils import validate_and_parse_decision
from probablyprofit.utils.validators import (
    validate_strategy,
    wrap_strategy_safely,
)


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
        strategy: Any | None = None,
        dry_run: bool = False,
    ):
        """
        Initialize OpenAI agent.
        """
        super().__init__(
            client, risk_manager, name, loop_interval, strategy=strategy, dry_run=dry_run
        )

        self.openai = OpenAI(api_key=openai_api_key)
        self.model = model

        # Validate and sanitize strategy prompt to prevent injection attacks
        sanitized_strategy, strategy_warnings = validate_strategy(strategy_prompt)
        self.strategy_prompt = wrap_strategy_safely(sanitized_strategy)

        # Surface strategy validation warnings to user
        if strategy_warnings:
            logger.warning("Strategy validation warnings:")
            for warning in strategy_warnings:
                logger.warning(f"  - {warning}")

        self.temperature = 0.7

        logger.info(f"OpenAIAgent '{name}' initialized with model {model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                ConnectionError,
                TimeoutError,
                NetworkException,
            )
        ),
    )
    async def _call_openai_api(self, api_kwargs: dict) -> str:
        """
        Call OpenAI API with retry logic.

        Args:
            api_kwargs: Arguments to pass to the API

        Returns:
            Response content text

        Raises:
            NetworkException: On connection/timeout errors (will be retried)
            AgentException: On non-retryable errors
        """
        try:
            response = await asyncio.to_thread(self.openai.chat.completions.create, **api_kwargs)

            if not response.choices or len(response.choices) == 0:
                raise AgentException("No response choices from OpenAI")

            return response.choices[0].message.content

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"OpenAI API connection error (will retry): {e}")
            raise NetworkException(f"OpenAI API connection error: {e}")
        except Exception as e:
            error_str = str(e).lower()
            if any(
                x in error_str for x in ["timeout", "connection", "rate limit", "429", "503", "502"]
            ):
                logger.warning(f"OpenAI API transient error (will retry): {e}")
                raise NetworkException(f"OpenAI API transient error: {e}")
            raise AgentException(f"OpenAI API error: {e}")

    def _format_observation(self, observation: Observation) -> str:
        """
        Format observation using shared formatter.

        Args:
            observation: Market observation

        Returns:
            Formatted prompt string
        """
        return ObservationFormatter.format_full_observation(
            observation, self.memory, include_history=5, max_markets=20
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(SchemaValidationError),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def decide(self, observation: Observation) -> Decision:
        """
        Use GPT-4 or o1 to make a trading decision with validation.

        Args:
            observation: Current market observation

        Returns:
            Decision based on AI analysis

        Raises:
            AgentException: If decision-making fails
        """
        logger.info(f"[{self.name}] Asking {self.model} for decision...")

        try:
            obs_text = self._format_observation(observation)

            # OpenAI 'o1' and 'o3' models have specific constraints:
            # 1. No 'system' role (use 'developer' or merge into 'user')
            # 2. No 'temperature' parameter (fixed at 1.0 usually)
            # 3. No 'response_format' json_object in early preview (sometimes)

            is_reasoning_model = self.model.startswith("o1") or self.model.startswith("o3")

            messages = []

            if is_reasoning_model:
                # For o1, we put everything in the user prompt or use developer role if available.
                # Currently safe bet is to prepend system instruction to user content.
                combined_prompt = f"""{self.strategy_prompt}

---
MARKET DATA:
{obs_text}

Respond in strict JSON format.
Output schema:
{get_decision_schema()}
"""
                messages.append({"role": "user", "content": combined_prompt})

                # Reasoning models don't support temperature
                kwargs = {}
                # Some o1 preview models support max_completion_tokens instead of max_tokens
            else:
                # Standard GPT-4 setup
                system_prompt = f"""{self.strategy_prompt}

You are a trading bot. Respond in strict JSON format.
Output schema:
{get_decision_schema()}
"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs_text},
                ]
                kwargs = {"temperature": self.temperature}

            # Call API
            # Note: o1/o3 reasoning models don't support response_format parameter
            api_kwargs = {"model": self.model, "messages": messages, **kwargs}

            # Only add response_format for non-reasoning models
            if not is_reasoning_model:
                api_kwargs["response_format"] = {"type": "json_object"}

            # Call API with retry logic
            content = await self._call_openai_api(api_kwargs)
            logger.debug(f"AI response: {content[:200]}...")

            # Validate and parse response using Pydantic schema
            decision = validate_and_parse_decision(content, Decision)

            # Additional clamping if Pydantic allowed out-of-bounds values (depending on model)
            if decision.confidence < 0 or decision.confidence > 1:
                logger.warning(f"Invalid confidence {decision.confidence}, clamping to 0-1")
                decision.confidence = max(0.0, min(1.0, decision.confidence))

            if decision.price is not None:
                if decision.price < 0 or decision.price > 1:
                    logger.warning(f"Invalid price {decision.price}, clamping to 0-1")
                    decision.price = max(0.0, min(1.0, decision.price))

            return decision

        except SchemaValidationError:
            # Re-raise SchemaValidationError to trigger retry
            logger.warning("Schema validation failed, will retry...")
            raise
        except AgentException:
            # Re-raise AgentException to propagate it up
            raise
        except Exception as e:
            logger.error(f"Error getting decision from OpenAI: {e}")
            raise AgentException(f"OpenAI decision error: {e}")
