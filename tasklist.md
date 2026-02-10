# API Enhancement: Category/Search Support for Polymarket Markets

## Overview
Extend the `get_markets()` method in `probablyprofit/api/client.py` to support category and search filtering via Polymarket Gamma API parameters. Add whitelist/blacklist keyword filtering loaded from `.env` variables for client-side filtering.

## Tasks

### 1. Research Polymarket Gamma API Parameters
- [ ] Investigate supported query parameters for `/markets` endpoint (e.g., `category`, `search`, `tags`)
- [ ] Check if API supports category-based filtering (e.g., `category=crypto`)
- [ ] Document available filters in code comments

### 2. Update Configuration (.env Support)
- [ ] Add new config fields in `probablyprofit/config.py`:
  - `market_whitelist_keywords`: List of keywords to include (e.g., "15M", "crypto")
  - `market_blacklist_keywords`: List of keywords to exclude
  - `market_category`: Optional category filter (e.g., "crypto")
  - `market_search`: Optional search term
- [ ] Load these from `.env` file with defaults

### 3. Extend get_markets() Method
- [ ] Modify `get_markets()` signature to accept new params:
  ```python
  async def get_markets(
      self,
      active: bool = True,
      limit: int = 100,
      offset: int = 0,
      category: Optional[str] = None,
      search: Optional[str] = None,
  ) -> List[Market]:
  ```
- [ ] Update `_get_markets_with_retry()` to pass category/search to API params
- [ ] Add client-side whitelist/blacklist filtering after API response

### 4. Update Base Agent Observe Logic
- [ ] Modify `observe()` in `probablyprofit/agent/base.py` to use config values:
  ```python
  markets = await self.client.get_markets(
      active=True,
      limit=cfg.api.market_fetch_limit,
      category=cfg.api.market_category,
      search=cfg.api.market_search,
  )
  ```
- [ ] Apply whitelist/blacklist filtering if configured

### 5. Update Strategy Classes
- [ ] Ensure `CustomStrategy` respects new config filters
- [ ] Add `CategoryStrategy` class for category-based trading

### 6. CLI Integration
- [ ] Add CLI options in `probablyprofit/cli/main.py` for category/search overrides
- [ ] Update help text and examples

### 7. Testing
- [ ] Add unit tests for new filtering logic
- [ ] Test with real API calls (dry-run mode)
- [ ] Verify whitelist/blacklist works with sample markets

### 8. Documentation
- [ ] Update `docs/api-reference.md` with new get_markets params
- [ ] Add examples in `docs/strategy-guide.md` for category targeting