# API Enhancement: Reliable Filtering for Polymarket Markets via Gamma API

## Overview
Redesign the `get_markets()` method in `probablyprofit/api/client.py` to enable efficient filtering of Polymarket markets, focusing on server-side support for categories via tags and client-side keyword filtering for specifics like "15 min" durations. This ensures reliable targeting of active 15-minute crypto markets by leveraging actual Gamma API parameters (e.g., `tag_id` for crypto) and post-fetch keyword checks. Load filters from `.env` for flexibility, with fallbacks for unsupported features.

## Tasks

### 1. Research Polymarket Gamma API Endpoints
- [ ] Review `/markets`, `/tags`, and `/events` endpoints for filtering options (e.g., `tag_id`, `closed`, `end_date_min/max`).
- [ ] Confirm tag-based filtering for categories (e.g., resolve "cryptocurrency" tag slug to ID for crypto markets).
- [ ] Identify patterns in market titles for client-side "15 min" filtering (e.g., "15M" in questions).
- [ ] Document supported params and limitations in code comments (e.g., no native `search` or `category` params).

### 2. Update Configuration (.env Support)
- [ ] Add new config fields in `probablyprofit/config.py`:
  - `market_whitelist_keywords`: Comma-separated keywords to include (e.g., "15M,15 min,BTC,ETH").
  - `market_blacklist_keywords`: Comma-separated keywords to exclude (e.g., "daily,weekly").
  - `market_tag_slug`: Optional tag slug for server-side filtering (e.g., "cryptocurrency").
  - `market_duration_max_minutes`: Optional max duration for markets (e.g., 15 for short-term filtering via dates).
- [ ] Load from `.env` with defaults (e.g., tag_slug="cryptocurrency", whitelist="15M").

### 3. Add get_tags() Method
- [ ] Implement new async method in `probablyprofit/api/client.py`:
  ```python
  async def get_tags(
      self,
      limit: int = 100,
      offset: int = 0,
  ) -> List[Dict[str, Any]]:
  ```
- [ ] Use `_get_with_retry("/tags")` to fetch tags.
- [ ] Add caching or optional refresh param to avoid repeated calls.

### 4. Extend get_markets() Method
- [ ] Update signature to use supported params:
  ```python
  async def get_markets(
      self,
      closed: bool = False,  # False for active markets
      limit: int = 100,
      offset: int = 0,
      tag_id: Optional[int] = None,
      end_date_max: Optional[str] = None,  # ISO datetime for max end date
  ) -> List[Market]:
  ```
- [ ] In `_get_markets_with_retry()`, pass valid API params (closed, limit, offset, tag_id, end_date_max).
- [ ] Pre-fetch logic: If tag_slug provided (from config), call `get_tags()` to resolve to tag_id; fallback to keyword if not found.
- [ ] Post-fetch: Apply client-side filtering – keep markets where `question` matches any whitelist keyword and none from blacklist.
- [ ] Handle full pagination: Loop fetches with increasing offset until no more results.

### 5. Update Base Agent Observe Logic
- [ ] In `probablyprofit/agent/base.py`, modify `observe()` to load config and resolve filters:
  ```python
  tag_id = await self._resolve_tag_id(cfg.api.market_tag_slug)  # Helper to get_tags() and map
  markets = await self.client.get_markets(
      closed=False,
      limit=cfg.api.market_fetch_limit,
      tag_id=tag_id,
      end_date_max=calculate_max_end_date(cfg.api.market_duration_max_minutes),  # Helper for now + duration
  )
  ```
- [ ] Integrate whitelist/blacklist directly in observe() or via get_markets().

### 6. Update Strategy Classes
- [ ] Modify `CustomStrategy` to use filtered markets from config (e.g., tag_slug and keywords).
- [ ] Add `ShortTermCryptoStrategy` class: Extends base with hardcoded tag_slug="cryptocurrency" and whitelist="15M".

### 7. CLI Integration
- [ ] In `probablyprofit/cli/main.py`, add options for overrides (e.g., --tag-slug, --whitelist, --duration-max).
- [ ] Update help: Include examples like "Target 15-min crypto: --tag-slug cryptocurrency --whitelist '15M,BTC'".
- [ ] Add dry-run flag to print filtered markets without trading.

### 8. Testing
- [ ] Write unit tests: Mock API responses for /tags and /markets; test tag resolution and keyword filtering.
- [ ] Integration tests: Real API calls in dry-run; verify only active crypto markets with "15M" are returned.
- [ ] Edge cases: Invalid tag_slug fallback, no matches, pagination over 100 markets.

### 9. Documentation
- [ ] Update `docs/api-reference.md`: Detail new methods/params (get_tags, get_markets with tag_id).
- [ ] Add `docs/filtering-guide.md`: Examples for 15-min crypto setup, tag resolution, and troubleshooting.