# Task List: Pydantic Schema Validation for LLM Responses

This task list outlines the phased approach to refactoring the agent codebases to enforce Pydantic schema validation for all LLM responses.

## Phase 1: Analysis & Design
- [x] Analyze existing agent implementations (`OpenAIAgent`, `AnthropicAgent`, `GeminiAgent`).
- [ ] Design a unified validation strategy using Pydantic's `model_validate_json` or `ValidationError` handling.
- [ ] Determine how to integrate validation failures into the existing `@retry` mechanism.
- [ ] Define the target "Strict" Decision schema if the current one needs adjustment for LLM consumption.

## Phase 2: Core Validation Infrastructure
- [ ] Create a generic `validate_and_parse_decision(json_str: str) -> Decision` utility function.
    - This function should handle `json.JSONDecodeError` and Pydantic `ValidationError`.
    - It should sanitize common LLM JSON quirks (e.g., markdown code blocks).
- [ ] Update `Decision` model if necessary to support specific validation rules (e.g., `Field` constraints for `confidence` between 0-1).

## Phase 3: Agent Refactoring
- [ ] **OpenAIAgent**:
    - [ ] Update `decide` method to use `Decision` schema for validation.
    - [ ] Leverage OpenAI's "Structured Outputs" (if library version allows) or enforce strict JSON schema in system prompt.
- [ ] **AnthropicAgent**:
    - [ ] Update `decide` method to parse response using the shared validation utility.
    - [ ] Refine prompt to ensure JSON output matches the Pydantic schema exactly.
- [ ] **GeminiAgent**:
    - [ ] Update `decide` method to use the validation utility.
    - [ ] Ensure `response_mime_type="application/json"` is effectively used with the schema.

## Phase 4: Reliability & Retries
- [ ] Implement a specific `SchemaValidationError` that can trigger the existing `@retry` decorator.
- [ ] (Optional) Modify retry logic to feed validation errors back to the LLM for self-correction (if feasible within current architecture).

## Phase 5: Testing & Deployment
- [ ] Create unit tests for `validate_and_parse_decision`.
    - Test valid JSON.
    - Test invalid JSON (syntax error).
    - Test valid JSON but invalid schema (wrong types, missing fields).
- [ ] Verify each agent against mock responses ensuring they retry on schema failures.
- [ ] Deploy changes.
