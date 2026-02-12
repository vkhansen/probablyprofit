# Filtering Guide

This guide explains how to configure `probablyprofit` to target specific types of markets using server-side and client-side filtering.

## Overview

Effective market filtering is crucial for a successful trading strategy. By narrowing down the universe of markets, you can focus your agent's attention on the opportunities that are most relevant to its logic.

The filtering is configured in your `.env` file or through CLI options and involves a combination of:

1.  **Server-Side Filtering**: Using the Polymarket Gamma API's built-in filtering capabilities (`tag_id`, `end_date_max`). This is the most efficient way to filter as it reduces the amount of data transferred.
2.  **Client-Side Filtering**: Applying keyword-based filters on the market questions after they have been fetched. This is useful for more specific filtering that the API doesn't support directly.

## Configuration

Filtering is controlled by the following configuration parameters (in your `.env` file or as CLI arguments):

-   `MARKET_TAG_SLUG`: The slug of the tag to filter by (e.g., "cryptocurrency"). The bot will resolve this to the correct `tag_id`.
-   `MARKET_WHITELIST_KEYWORDS`: A comma-separated list of keywords. Markets must contain at least one of these keywords in their question to be included.
-   `MARKET_BLACKLIST_KEYWORDS`: A comma-separated list of keywords. Markets containing any of these keywords will be excluded.
-   `MARKET_DURATION_MAX_MINUTES`: An integer representing the maximum duration of markets to trade, in minutes.

### Example: Targeting 15-Minute Crypto Markets

To configure the bot to trade only 15-minute cryptocurrency markets, you would set the following in your `.env` file:

```
MARKET_TAG_SLUG="cryptocurrency"
MARKET_WHITELIST_KEYWORDS="15M,15 min"
MARKET_BLACKLIST_KEYWORDS="daily,weekly"
MARKET_DURATION_MAX_MINUTES=20
```

Or, using CLI arguments:

```bash
probablyprofit run "My crypto strategy" \
    --tag-slug cryptocurrency \
    --whitelist "15M,15 min" \
    --duration-max 20
```

This configuration will:
1.  Fetch only markets tagged with "cryptocurrency".
2.  Only consider markets that will resolve in the next 20 minutes.
3.  From that list, only include markets with "15M" or "15 min" in their title.
4.  Exclude any markets that contain the words "daily" or "weekly".

This layered approach ensures you are targeting a very specific subset of markets, which is ideal for a specialized strategy like the `ShortTermCryptoStrategy`.
