# US Sharadar External Data Requirements and Next-Step Plan

Date: 2026-05-11

This document summarizes the current state of the US Sharadar/Qlib trading-model pipeline, the problems that are still blocking a release-grade model, the specific new data we should search for, and the implementation plan after a data source is selected.

The practical goal is not just to add more data. The goal is to add data that directly addresses the current failure modes: weak forward-looking alpha, bad top-bucket selection in adverse regimes, low annualized excess return versus QQQ/SPY/IXIC, and unstable year-by-year performance.

This is research infrastructure, not a guarantee of trading profit. A model should only be considered for live use after it passes the repo's strict release gates on current point-in-time data, survives realistic trading-cost stress, and performs robustly against QQQ, SPY, and IXIC.

## Executive Summary

The current pipeline is much healthier than it was mechanically. The latest validation logs show that the main integrity controls are now working:

- Train/valid/test ordering passes.
- Embargoes pass.
- Prediction coverage passes.
- Walk-forward manifests pass.
- SF1/SF3A/model-feature provenance passes.
- QQQ/SPY/IXIC benchmark wiring is in place.
- Current validation includes strict model-quality checks, rebalance-date checks, active-risk checks, rolling checks, stress-cost checks, and multiple-testing tracking.

The remaining bottleneck is signal quality, not basic pipeline mechanics. The latest candidates are still not usable for live trading because the model cannot reliably identify a top-ranked basket that beats strong benchmarks across regimes.

The most important next data to add is:

1. Analyst expectations and revisions.
2. Earnings calendar, confirmed report timing, and surprise history.
3. Properly engineered insider activity from the existing Sharadar SF2 data.
4. Better institutional/13F change features from the existing Sharadar SF3A data.
5. SEC filing event metadata and selected filing text signals.
6. Short interest, borrow pressure, and crowding data if available at reasonable cost.
7. Options/implied-volatility features later, mainly for event risk and crowding.
8. News/key-development data only after cleaner structured event data is working.

The best first paid external data pilot is a low-cost expectations/calendar source, such as FMP or EODHD. If budget allows and licensing matters for serious deployment, Intrinio/Zacks-style estimate data is likely higher quality but quote-based and more expensive.

## Current Local Data Inventory

The current local Sharadar raw store includes:

| Dataset | Local coverage observed | Main contents | Current role |
|---|---:|---|---|
| `SEP` | 1401 files | US equity daily prices | Price/return history and universe support |
| `SFP` | 21 files | Fund/ETF prices | Benchmark and ETF context |
| `SF1_MRQ` | 1 consolidated file | Point-in-time fundamentals and ratios | Core fundamental features |
| `SF2` | 1135 files | Insider transactions | Available locally but still underused as predictive features |
| `SF3A` | 1137 files | Institutional holder aggregates | Used with lag correction, but can be expanded |
| `tickers.csv` | 1 file | Ticker metadata | Mapping and universe support |

Important local columns:

- `SF1_MRQ` includes `ticker`, `dimension`, `calendardate`, `datekey`, `reportperiod`, `lastupdated`, balance sheet items, cash flow items, valuation ratios, market cap, revenue, margins, shares, price, and many other fields.
- `SF2` includes `filingdate`, `formtype`, `ownername`, `officertitle`, director/officer/10-percent-owner flags, `transactiondate`, `transactioncode`, `transactionshares`, `transactionpricepershare`, `transactionvalue`, and ownership fields.
- `SF3A` includes `calendardate`, `ticker`, holder counts, call/put holders, share/call/put units, values, total value, and percent of total.

Sharadar remains valuable. It gives us clean US public-company fundamentals, survivorship-aware price history, insiders, institutional holdings, and ETF/fund prices. The issue is that the current model mostly sees stale accounting state and historical price/risk features. It does not yet see enough market expectations, forward revisions, or event timing.

## Current Candidate Failures

The latest experiments improved diagnostics but did not produce a release candidate.

### V4 IC-Selected Score Model

Artifact:

- `artifacts/research/score_growth_v4_icselect_single_20260510/screened/`

Observed model-quality results:

| Metric | Result |
|---|---:|
| Full mean IC | `-0.0069` |
| Full top-quantile minus bottom-quantile spread | `-0.0122` |
| Recent 63d mean IC | `0.0533` |
| Recent 63d top-k label | `-0.0042` |
| Positive IC years | `2/4` eligible years |
| Worst-year mean IC | `-0.0575` |
| Worst-year top-k label | `-0.0368` |
| Worst-year top-quantile spread | `-0.0307` |

Year details:

| Year | Mean IC | Top-k label | Top-quantile spread |
|---|---:|---:|---:|
| 2022 | `-0.0575` | `-0.0368` | `-0.0307` |
| 2023 | `0.0366` | `-0.0089` | `0.0036` |
| 2024 | `0.0044` | `0.0250` | `-0.0015` |
| 2025 | `-0.0216` | `-0.0154` | `-0.0269` |
| 2026 | `0.0545` | `-0.0248` | `0.0258` |

Interpretation:

- This model is not release-usable.
- It can show recent IC strength while still selecting a bad trade basket.
- The top-ranked names are not consistently the names we want to own.
- Recent rank correlation alone is not enough; top-bucket realized returns matter.

### V5 Regime-Aware IC-Selected Score Model

Artifact:

- `artifacts/research/score_growth_v5_regimeic_single_20260511/screened/`

Observed model-quality results:

| Metric | Result |
|---|---:|
| Full mean IC | `0.0136` |
| Full top-quantile minus bottom-quantile spread | `-0.0051` |
| Recent 63d mean IC | `0.0307` |
| Recent 63d top-k label | `0.0615` |
| Positive IC years | `3/4` eligible years |
| Worst-year mean IC | `-0.0304` |
| Worst-year top-k label | `-0.0168` |
| Worst-year top-quantile spread | `-0.0215` |

Year details:

| Year | Mean IC | Top-k label | Top-quantile spread |
|---|---:|---:|---:|
| 2022 | `-0.0304` | `-0.0069` | `-0.0215` |
| 2023 | `0.0242` | `-0.0168` | `-0.0053` |
| 2024 | `0.0362` | `0.0349` | `0.0094` |
| 2025 | `0.0233` | `0.0321` | `-0.0080` |
| 2026 | `0.0190` | `0.0438` | `0.0242` |

Interpretation:

- V5 is the best recent direction, but it still fails strict release-quality checks.
- Full mean IC is positive, recent performance is encouraging, and positive-IC-year count passes.
- However, the full top-quantile spread is still negative.
- 2022 remains bad, and 2023 top-k is negative even though overall IC is positive.
- This means the model has some ranking signal, but the investable top bucket is still not robust.

### V6 Tail-Regime IC-Selected Score Model

Artifact:

- `artifacts/research/score_growth_v6_tailregimeic_single_20260511/screened/`

Observed model-quality results:

| Metric | Result |
|---|---:|
| Full mean IC | `0.0061` |
| Full top-quantile minus bottom-quantile spread | `-0.0067` |
| Recent 63d mean IC | `0.0323` |
| Recent 63d top-k label | `0.0900` |
| Positive IC years | `3/4` eligible years |
| Worst-year mean IC | `-0.0335` |
| Worst-year top-k label | `-0.0262` |
| Worst-year top-quantile spread | `-0.0240` |

Year details:

| Year | Mean IC | Top-k label | Top-quantile spread |
|---|---:|---:|---:|
| 2022 | `-0.0335` | `-0.0199` | `-0.0240` |
| 2023 | `0.0130` | `-0.0262` | `-0.0043` |
| 2024 | `0.0183` | `0.0325` | `0.0021` |
| 2025 | `0.0230` | `0.0424` | `-0.0072` |
| 2026 | `0.0260` | `0.0523` | `0.0302` |

Interpretation:

- Tail-aware scoring did not fix the adverse-regime top-bucket problem.
- It reduced full mean IC versus V5.
- It improved recent top-k results, but that is not enough for release.
- It confirms that ad hoc tail weighting is not the missing ingredient by itself.

## Main Problems We Need to Solve

### 1. The Top Basket Is Not Reliable

The model can sometimes rank the cross-section in a way that produces positive IC, but the actual top-k or top-quantile bucket can still be negative. For trading, this is the key failure. We do not get paid for a mild rank correlation if the selected portfolio does not make enough money after costs and benchmark drag.

Evidence:

- V5 full mean IC passed, but full top-quantile spread failed.
- V5 2023 mean IC was positive, but top-k label was negative.
- V6 recent metrics looked strong, but full and worst-year top-bucket checks still failed.

What this implies:

- The objective should care directly about investable top-bucket behavior, not only average rank IC.
- We need features that identify actual upside catalysts and avoid crowded downside traps.
- More strategy/risk tuning will not fix a bad candidate set.

### 2. The Model Is Fragile Across Regimes

The model performs materially worse in adverse or changing regimes. 2022 remains the clearest failure period, and 2023 still shows weak top-k selection in recent candidates.

Likely causes:

- Training features overrepresent normal growth/momentum behavior.
- Existing market regime features are too coarse.
- The model does not understand earnings expectations, estimate cuts, inflation/rate sensitivity, financing risk, short-crowding, or event-driven repricing.
- Static fundamentals update too slowly to explain sudden changes in forward expectations.

What this implies:

- We need data that changes when expectations change, not only after fundamentals are reported.
- We should add regime-conditioned feature screening and event-aware targets, but only after the raw signal set is stronger.

### 3. Current Sharadar-Only Features Are Too Backward-Looking

SF1 fundamentals are useful, but many fields are accounting facts. Even with correct point-in-time handling, they are not enough to capture:

- Analyst estimate revisions.
- Earnings surprise expectations.
- Confirmed earnings announcement timing.
- Guidance changes.
- Investor-positioning stress.
- Options-implied event risk.
- News/key-development catalysts.

What this implies:

- Sharadar is a strong base layer, but not sufficient as currently used for a high-return, benchmark-beating growth model.
- We should fully exploit SF2/SF3A first because we already have them, but the largest missing category is forward expectations.

### 4. Annualized Excess Return Is Too Low Versus QQQ/SPY/IXIC

The user goal requires beating strong baselines such as QQQ, SPY, and IXIC. A model with low active return is not useful, even if it has acceptable turnover or a decent raw information coefficient.

Observed pattern:

- Risk controls can reduce drawdown but often erase alpha.
- Benchmark-core overlays can stabilize exposure, but they cannot create alpha.
- Strategy-only tuning on fixed predictions previously made some variants worse.

What this implies:

- We need better candidate generation, not looser gates.
- We need higher expected active return per trade, stronger catalyst signals, and cleaner avoidance of bad top picks.

### 5. There Is Still Overfitting and Multiple-Testing Risk

The repo now tracks trials and validates with stricter checks, but the research process has tested many candidates. This creates selection bias risk.

What this implies:

- Any new data source must be evaluated by a pre-registered experiment set.
- We should avoid buying a dataset, trying hundreds of combinations, and declaring victory on one lucky backtest.
- We need an explicit stop/pivot rule.

### 6. Existing SF2 and SF3A Are Not Fully Exploited Yet

SF2 insider transactions and SF3A institutional holdings are already local. They should be engineered more directly before or alongside external data.

Potential SF2 features:

- Net insider buy value over 20/63/126 trading days.
- Open-market buy value using transaction code `P`.
- Open-market sell value using transaction code `S`.
- Buy/sell imbalance by dollar value and share count.
- Cluster-buy count across distinct insiders.
- Officer/director buy indicators.
- CEO/CFO purchase flags if title parsing is reliable enough.
- Net purchase value scaled by market cap.
- Repeat buyer/seller streaks.
- Large purchase percentile relative to the issuer's own history.
- Lag by `filingdate`, not `transactiondate`.

Potential SF3A features:

- Change in number of share holders.
- Change in share value and share units.
- Call/put holder imbalance.
- Put/call value ratio.
- Institutional concentration proxy.
- Quarter-over-quarter institutional accumulation.
- 13F snapshot age.
- Features using a 45-calendar-day availability lag, not raw `calendardate`.

## New Data We Should Search For

The categories below are ordered by expected value for the current weekly/monthly growth pipeline.

### Priority 1: Analyst Estimates, Revisions, Ratings, and Price Targets

This is the highest-priority external data category.

Fields to look for:

- Consensus EPS estimates for next quarter, current fiscal year, next fiscal year.
- Consensus revenue estimates for next quarter, current fiscal year, next fiscal year.
- Historical estimate snapshots, not just latest values.
- Revision counts: up/down revisions over 7, 30, 60, and 90 days.
- Revision magnitude: change in consensus EPS/revenue estimates over 7, 30, 60, and 90 days.
- Analyst count.
- Estimate dispersion or standard deviation.
- Price target consensus, high/low/median/mean.
- Price target revisions.
- Ratings distribution: strong buy, buy, hold, sell, strong sell.
- Rating upgrades/downgrades with action date.
- Long-term growth estimates if available.
- Surprise history and whether the provider stores both estimate and actual as of the event.
- Exact effective date or publication date for each snapshot/revision.

Why this helps:

- Current SF1 fundamentals show what happened. Estimate revisions show what the market is beginning to expect.
- Growth stocks are often priced on forward revenue/EPS trajectories, not trailing accounting state.
- Negative estimate revisions can help avoid value traps and growth deceleration.
- Positive revisions after earnings can support post-earnings drift targets.

Candidate features:

- `eps_est_revision_30d`
- `revenue_est_revision_30d`
- `eps_est_revision_accel_30d_vs_90d`
- `revenue_est_revision_accel_30d_vs_90d`
- `rating_upgrade_count_30d`
- `rating_downgrade_count_30d`
- `price_target_revision_30d`
- `price_target_upside_to_price`
- `estimate_dispersion`
- `analyst_count_change_90d`
- `revision_breadth = up_revisions / (up_revisions + down_revisions)`
- `forward_sales_growth_est`
- `forward_eps_growth_est`
- `surprise_streak`

Point-in-time requirements:

- Historical snapshots must be available as they were known on each date.
- The provider must not overwrite old consensus values with revised current values.
- Dates must reflect when the estimate/revision became available.
- If only event-date actuals are available, we must lag them to after announcement time.

Vendor search terms:

- "historical analyst estimates API point in time"
- "EPS estimate revisions API historical snapshots"
- "revenue estimate revisions API"
- "analyst ratings upgrades downgrades historical API"
- "price target consensus historical API"
- "Zacks estimates API"
- "earnings estimates point in time dataset"

### Priority 2: Earnings Calendar, Confirmed Report Time, and Surprise Data

Fields to look for:

- Historical earnings report date.
- Confirmed upcoming earnings report date.
- Announcement timing: before market open, after market close, during market hours, unknown.
- EPS estimate before the release.
- EPS actual after the release.
- Revenue estimate before the release.
- Revenue actual after the release.
- EPS surprise percentage.
- Revenue surprise percentage.
- Guidance fields if available.
- Preliminary versus confirmed date changes.
- Restatement/correction markers if available.

Why this helps:

- Earnings are the dominant scheduled catalyst for many stocks.
- The current weekly pipeline can accidentally hold through events without knowing event risk.
- Event-aware targets can separate pre-earnings drift, earnings gap, and post-earnings drift.
- Confirmed earnings timing lets us avoid lookahead around actual EPS/revenue.

Candidate features:

- `days_to_earnings`
- `days_since_earnings`
- `earnings_window_flag`
- `eps_surprise_last_q`
- `revenue_surprise_last_q`
- `surprise_streak_4q`
- `post_earnings_drift_5d`
- `post_earnings_drift_20d`
- `pre_earnings_runup_20d`
- `estimate_revision_pre_earnings_30d`
- `earnings_time_bmo_amc`

Point-in-time requirements:

- Actual EPS/revenue cannot be visible before the release time.
- For after-market releases, earliest tradable use should normally be the next trading day.
- For before-market releases, same-day use may be possible only if the pipeline explicitly supports pre-open data availability. The current conservative default should be next trading day.

Vendor search terms:

- "earnings calendar API confirmed before after market"
- "historical earnings surprise API EPS revenue"
- "earnings report time API BMO AMC"
- "earnings trend API"

### Priority 3: Insider Trading Signals From Existing Sharadar SF2

We already have this data. We should build it regardless of external vendor choice.

Fields already present:

- `filingdate`
- `transactiondate`
- `transactioncode`
- `transactionshares`
- `transactionpricepershare`
- `transactionvalue`
- director/officer/10-percent-owner flags
- `officertitle`
- shares owned before/after transaction

Why this helps:

- Open-market insider purchases can be informative, especially when clustered.
- Insider sales alone are noisy, but extreme selling after price spikes can help with risk.
- Insider activity can complement valuation/growth screens.

Candidate features:

- Net insider purchase value over 20/63/126 trading days.
- Open-market purchase count over 20/63/126 trading days.
- Distinct buyer count over 20/63/126 trading days.
- Distinct seller count over 20/63/126 trading days.
- CEO/CFO purchase flag over 126 trading days.
- Director purchase flag over 126 trading days.
- 10-percent-owner purchase flag.
- Purchase value scaled by market cap.
- Purchase value scaled by average daily dollar volume.
- Cluster-buy flag: at least 2 or 3 distinct insiders buying within 30 days.
- Large-buy percentile relative to the issuer's own trailing history.

Point-in-time requirements:

- Availability date must be `filingdate`.
- `transactiondate` should be used only as event context, not as availability.
- Some transaction codes should be excluded or separated:
  - `P`: open-market purchase, usually highest signal.
  - `S`: sale, noisy but useful in aggregate.
  - Option exercise, award, grant, gift, and derivative transactions should not be treated the same as open-market buying.

### Priority 4: Institutional Ownership and 13F Changes From Existing Sharadar SF3A

We already have this data, but the feature set should be expanded.

Fields already present:

- `shrholders`, `cllholders`, `putholders`
- `shrunits`, `cllunits`, `putunits`
- `shrvalue`, `cllvalue`, `putvalue`
- `totalvalue`, `percentoftotal`

Why this helps:

- Institutional accumulation or distribution can explain medium-horizon relative strength.
- Put/call institutional exposure can provide crowding/risk information.
- Holder-count changes can detect broadening or narrowing sponsorship.

Candidate features:

- Quarter-over-quarter change in holder count.
- Quarter-over-quarter change in share value.
- Quarter-over-quarter change in share units.
- Share-holder count acceleration.
- Put/call holder ratio.
- Put/call value ratio.
- Call value as percentage of total institutional value.
- Put value as percentage of total institutional value.
- Institutional ownership concentration proxy.
- `percentoftotal` change.
- Snapshot age and stale-snapshot flag.

Point-in-time requirements:

- Use the repo's 45-calendar-day availability lag.
- Do not use `calendardate` as if known immediately.

### Priority 5: SEC Filing Events and Filing Text

SEC EDGAR data is free and should be considered a strong supplement.

Useful events:

- 10-K and 10-Q accepted timestamps.
- 8-K accepted timestamps.
- 13D/13G activist or large-holder filings.
- S-3 shelf registrations.
- Equity offering announcements.
- Form 4 insider filings as a cross-check to SF2.
- NT 10-K/10-Q late filing notices.
- Auditor changes, restatements, investigations, and material agreements from 8-K item metadata.

Why this helps:

- Some important catalysts are not captured by structured fundamentals.
- Filing timing itself can be predictive.
- Negative filing events can help avoid catastrophic top-basket picks.

Candidate features:

- `days_since_10q`
- `days_since_10k`
- `days_since_8k`
- 8-K item category flags.
- Late-filing flag.
- Offering/shelf filing flag.
- Activist filing flag.
- Text-based risk-change score for 10-Q/10-K sections.
- Similarity/change score versus prior filing.

Point-in-time requirements:

- Use SEC accepted datetime, not report period.
- Text features must be generated only from filings available at that datetime.
- For daily strategy, conservative availability should be next trading day after accepted datetime unless the pipeline supports intraday release timing.

Source note:

- The SEC states that `data.sec.gov` APIs do not require authentication or API keys and provide JSON-formatted submissions and XBRL data. Programmatic access must comply with SEC policies.

What we need from the user:

- A real SEC User-Agent contact string, ideally `ProjectName contact@email.com`.

### Priority 6: Short Interest, Borrow, and Crowding

Fields to look for:

- Official short interest shares.
- Short interest as percentage of float.
- Days to cover.
- Settlement date and publication date.
- Securities lending utilization.
- Borrow fee.
- Recall pressure if available.
- Fail-to-deliver data if available.

Why this helps:

- Crowded shorts can rally violently and break naive short/underweight assumptions.
- High short interest plus positive estimate revisions can be a powerful long catalyst.
- High borrow pressure plus deteriorating fundamentals can identify weak names.

Candidate features:

- `short_interest_pct_float`
- `short_interest_change_2w`
- `days_to_cover`
- `borrow_fee`
- `borrow_fee_change_20d`
- `utilization`
- `short_squeeze_risk`
- `revision_x_short_interest_interaction`

Point-in-time requirements:

- Short interest has settlement dates and publication dates. Use publication/availability date.
- Do not use settlement date as if the data was known then.

Cost note:

- FINRA public API credentials may provide access to public data at no cost with usage limits, but production firm/organization credentials are listed by FINRA at about `$1,650/month` plus overage fees. Some exchange or vendor feeds may be cheaper for limited short-interest use.

### Priority 7: Options and Implied Volatility

This is valuable, but it should not be the first paid data source unless we decide to pivot into event or higher-frequency trading.

Fields to look for:

- Option chain snapshots.
- ATM implied volatility.
- IV rank/percentile.
- Skew: put IV minus call IV, 25-delta skew, or similar.
- Term structure: front-month IV versus later-month IV.
- Open interest by strike/expiry.
- Volume by strike/expiry.
- Put/call volume and open-interest ratios.
- Earnings implied move.
- Dealer gamma exposure if available, though this is often vendor-derived and harder to audit.

Why this helps:

- Options data can identify event risk and market-implied uncertainty.
- It can help avoid names with expensive or dangerous earnings risk.
- It can improve position sizing around earnings.

Candidate features:

- `atm_iv_30d`
- `iv_rank_252d`
- `put_call_oi_ratio`
- `put_call_volume_ratio`
- `skew_30d`
- `term_structure_slope`
- `earnings_implied_move`
- `option_volume_zscore`

Point-in-time requirements:

- Snapshot timestamp matters.
- For daily/weekly models, end-of-day option snapshots are usually enough.
- Corporate actions and option-symbol changes must be handled carefully.

Cost note:

- Massive/Polygon options plans currently show individual options tiers around free, `$29/month`, `$79/month`, and `$199/month`, with business options around `$1,999/month`.
- ThetaData currently shows options plans around `$40/month`, `$80/month`, and `$160/month` for individual use.
- Licensing differs sharply between individual research and commercial/business use.

### Priority 8: News, Sentiment, and Key Developments

This can help, but it is noisier and harder to validate than estimates/calendar data.

Fields to look for:

- Timestamped news articles.
- Company-event categories.
- Analyst action news.
- M&A, product, regulatory, lawsuit, management-change, FDA, contract, guidance, offering, and bankruptcy tags.
- Sentiment scores with historical timestamps.
- Source reliability.
- Deduplication/grouping IDs.

Why this helps:

- It can detect catalysts not present in structured data.
- It may help avoid major negative events.

Risks:

- Sentiment scores can be unstable and vendor-specific.
- News data can cause heavy multiple-testing risk.
- Full text licensing can be expensive.
- Some APIs provide only recent history, which is not enough for robust backtesting.

Recommendation:

- Do not start here unless the source includes clean event categories and enough history.
- Prefer structured key developments over raw article sentiment for the first implementation.

### Priority 9: Macro, Rates, Credit, and Sector Regime Data

Fields to look for:

- Treasury yields and yield-curve changes.
- Credit spreads.
- Inflation expectations.
- Fed funds expectations if available.
- VIX and volatility indices.
- Sector ETF returns, breadth, and drawdowns.
- Dollar index, oil, and commodity proxies.
- Liquidity proxies.

Why this helps:

- 2022-style regime failures likely need better macro/rate context.
- Growth models are sensitive to rates, liquidity, and risk appetite.

Recommendation:

- Many macro series are free or cheap.
- This should be added as gating/regime context, not as a replacement for security-level alpha data.

## Provider Shortlist for the User to Research

The table below is not a final procurement decision. Prices and entitlements can change, and licensing terms matter. Use this as a search guide and verify the exact plan before buying.

| Provider/source | Best use | Approximate cost observed 2026-05-11 | Strength | Main concern |
|---|---|---:|---|---|
| Existing Sharadar | Fundamentals, prices, insiders, institutional holdings | Already available | Strong base layer, survivorship-aware US data | Not enough expectations/event data as currently used |
| SEC EDGAR | Filings, accepted timestamps, XBRL, filing events | Free | Primary source, no API key | Requires parsing, rate policy compliance, mapping work |
| FMP | Analyst estimates, earnings calendar, ratings, price targets, broad API | Premium around `$59/month` billed annually, Ultimate around `$149/month` billed annually | Cheapest practical first paid pilot | Need to verify point-in-time historical snapshots and license terms |
| EODHD | Earnings calendar/trends, fundamentals, news, broad API | Calendar/news around `$19.99/month`, fundamentals around `$59.99/month`, All-in-One around `$99.99/month` | Good broad low-cost alternative | Need to verify analyst-revision depth and PIT behavior |
| Intrinio/Zacks | EPS/sales estimates, surprises, ratings, target prices | Estimate datasets often quote-based; some Intrinio products list higher annual pricing | Likely better institutional-grade estimate history | Higher cost and commercial licensing complexity |
| FINRA | Official short interest/public regulatory data | Public credentials can be free with limits; organization/firm credentials around `$1,650/month` | Official source | Cost and entitlement complexity for production access |
| Massive/Polygon | Options, market data, IV/Greeks/open interest | Individual options tiers around `$29`, `$79`, `$199/month`; business options around `$1,999/month` | Developer-friendly options data | More useful after event strategy design; license restrictions |
| ThetaData | Options history/snapshots/streams | Around `$40`, `$80`, `$160/month` for individual options tiers | Strong options research candidate | Options are not the first missing alpha layer |
| Benzinga/RavenPack/AlphaSense/etc. | News/key developments/sentiment | Often sales-led or tiered | Useful for event tags | Expensive/noisy; historical licensing matters |

Official reference links checked:

- Nasdaq Data Link / Sharadar overview: https://www.nasdaq.com/nasdaq-data-link
- Sharadar data overview: https://www.sharadar.com/data
- FMP pricing: https://intelligence.financialmodelingprep.com/pricing-plans?direct=true
- FMP analyst estimates docs: https://site.financialmodelingprep.com/developer/docs/analyst-estimates-api
- FMP earnings calendar docs: https://site.financialmodelingprep.com/developer/docs/stable/earnings-calendar
- EODHD pricing: https://eodhd.com/pricing
- EODHD calendar/earnings docs: https://eodhd.com/knowledgebase/calendar-upcoming-earnings-ipos-and-splits/
- Intrinio pricing: https://intrinio.com/pricing
- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- FINRA API fees: https://developer.finra.org/fees
- FINRA support fee summary: https://developer.finra.org/node/241
- Massive/Polygon options pricing: https://polygon.io/pricing?product=options
- ThetaData pricing: https://www.thetadata.net/pricing

## Vendor Due-Diligence Checklist

Before buying or integrating a provider, verify the items below.

### Data History and Coverage

- Does it cover US equities broadly, not only mega caps?
- Does it include delisted symbols or historical inactive symbols?
- How far back does the dataset go?
- Does it cover at least 2016 to present? Ideally 2008 to present.
- Does it include small and mid caps, or only liquid large caps?
- Does it include ETFs where needed?
- Does it have stable identifiers: ticker history, CIK, CUSIP, FIGI, exchange, or permanent security ID?

### Point-in-Time Safety

- Are historical estimates stored as snapshots as of each date?
- Does the API return old consensus estimates, or only latest overwritten values?
- For earnings data, when did the estimate and actual become known?
- Are report dates timestamped before market/after market?
- Are filing/event timestamps available in UTC or exchange-local time?
- Are corrections/restatements marked?
- Can we reconstruct what was known on any historical trading date?

### Delivery and Scalability

- Is there a bulk endpoint, flat-file download, or S3 delivery?
- Can we pull 1000 to 2000 symbols without hitting impossible rate limits?
- Are API calls charged per symbol, per row, or per request?
- Can historical backfill be downloaded efficiently?
- Is CSV/Parquet available, or only JSON?
- Are there clear error codes and retry rules?

### Licensing

- Is personal research allowed?
- Is live personal trading allowed?
- Is business/commercial use allowed?
- Is machine-learning model training allowed?
- Can derived features and model scores be stored permanently?
- Can data be redistributed or displayed? If not, does that matter for our use?
- Are there exchange fees or non-professional requirements?
- Does the license survive cancellation for historical research artifacts?

### Data Quality

- Are sample files available?
- Are split/corporate-action adjustments clear?
- Are timestamps consistent?
- Are duplicate rows common?
- Are missing values documented?
- Are restatements handled?
- Are vendor-derived fields explainable?
- Does the provider disclose update schedules?

## What the User Should Provide After Searching

For each candidate API/data source, provide:

1. Provider name and plan name.
2. Monthly or annual cost.
3. Whether the intended use is personal research, personal live trading, or commercial/business use.
4. Links to endpoint documentation.
5. A sample response or CSV sample for at least:
   - analyst estimates or revisions,
   - earnings calendar/surprises,
   - ratings/price targets if included.
6. Coverage start date.
7. Whether historical point-in-time snapshots are included.
8. Rate limits and bulk download support.
9. Symbol identifier fields.
10. License terms for ML/backtesting/live trading.
11. API key after purchase, if you want me to implement against it.
12. SEC User-Agent email/name if we add SEC EDGAR collection.

Minimum sample tickers to test:

- `AAPL`
- `MSFT`
- `NVDA`
- `TSLA`
- `META`
- `JPM`
- `XOM`
- `COST`
- `AVGO`
- A few smaller or recently listed names.
- A few delisted or renamed names if the provider supports them.

## Recommended Data Acquisition Order

### Step 1: Finish Existing Sharadar SF2/SF3A Alpha Extraction

Do this even before selecting a paid vendor.

Deliverables:

- SF2 insider feature builder.
- Expanded SF3A institutional feature builder.
- Provenance metadata.
- PIT availability checks.
- Synthetic tests for transaction-code handling and `filingdate` lag.
- Small ablation configs: baseline, SF2 only, SF3A expanded only, combined.

Success criteria:

- Features are generated for the same investable universe.
- No availability-date leakage.
- Added features improve top-bucket diagnostics or at least do not degrade them.
- Validation still passes data-quality and provenance gates.

### Step 2: Add Analyst Estimates and Earnings Calendar

This is the highest-value external pilot.

Recommended default:

- Start with FMP or EODHD for a low-cost one-month pilot.
- If the provider cannot supply historical point-in-time estimate snapshots, use it only for limited event/calendar features and keep looking.
- If budget allows, compare against Intrinio/Zacks-style data for quality.

Deliverables:

- External raw downloader.
- Append-only raw storage.
- Metadata/provenance JSON per pull.
- PIT feature builder.
- Endpoint-specific tests using stored sample responses.
- Coverage report by ticker/date.
- Missingness report.
- Vendor-quality audit report.

Candidate features:

- Estimate revisions.
- Analyst count/coverage changes.
- Estimate dispersion.
- Ratings and price-target changes.
- Earnings date and report-time flags.
- Surprise history and post-earnings drift context.

Success criteria:

- Provider gives enough history for 2022 to 2026 validation.
- Features can be aligned without lookahead.
- New feature groups improve out-of-sample top-k/top-quantile behavior.
- Improvements survive ablation and multiple-testing controls.

### Step 3: Redesign Targets Around Catalysts and Active Return

The current generic forward-return target may be too broad. After estimates/earnings data arrives, build target families that are closer to how the model should make money.

Target families:

- QQQ-excess return over 5/10/20/40 trading days.
- SPY-excess return over 5/10/20/40 trading days.
- IXIC-excess return over 5/10/20/40 trading days.
- Volatility-scaled excess return.
- Top-vs-bottom classification target.
- Event-conditioned post-earnings drift target.
- Pre-earnings avoidance/risk target.
- Drawdown-aware utility target.

Design principle:

- The model should learn "what should we own next" under realistic selection and cost constraints, not only "which stocks have slightly higher continuous forward returns."

### Step 4: Build Candidate Families With Explicit Ablations

Do not run an unconstrained model zoo. Pre-register a small controlled candidate set.

Candidate families:

- Baseline current best Sharadar-only candidate.
- Sharadar plus SF2.
- Sharadar plus expanded SF3A.
- Sharadar plus estimates/calendar.
- Sharadar plus SF2/SF3A plus estimates/calendar.
- Event-specific model for post-earnings drift.
- Regime-gated ensemble of broad model plus event model.

Models:

- Start with LightGBM rank/regression/classification variants because the current pipeline already supports them.
- Keep `ICSelectedScoreModel` as a transparent diagnostic baseline, not the only production direction.
- Consider neural/tabular models only after the feature set shows real signal in simpler models.
- GPU is available, but compute is not the current bottleneck. Alpha quality is.

### Step 5: Validate With Strict Release Gates

Every serious candidate must run through the release harness:

- Embargoed walk-forward training.
- Strict data-quality checks.
- PIT/provenance checks.
- Model-quality checks.
- Rebalance-date model-quality checks.
- Strategy-weighted quality checks.
- Active-risk checks.
- Baseline gates versus `QQQ,SPY,IXIC`.
- Stress execution with higher costs and open-price assumptions.
- Rolling deterioration checks.
- Multiple-testing trial registry.

Keep the current gates strict. Relaxing gates would only hide the problem.

### Step 6: Decide Go/No-Go Objectively

A candidate is not release-grade unless it satisfies all of the following:

- Passes strict release gates.
- Beats QQQ, SPY, and IXIC in the configured baseline checks.
- Has positive and meaningful full top-k/top-quantile behavior.
- Has no catastrophic year like current 2022 failures.
- Has acceptable drawdown and stress-cost performance.
- Has stable recent performance without relying only on the last 63 days.
- Has feature provenance and reproducible raw data.
- Has a clearly documented data license.

If the new expectations/calendar data does not materially improve top-bucket selection after a controlled pilot, we should pivot the strategy design rather than keep tuning the same weekly growth ranking pipeline.

## Possible Strategy Pivots if the Current Road Still Fails

If new data does not rescue the current approach, the next pivots should be deliberate.

### Pivot A: Event-Driven Earnings Drift

Instead of ranking all stocks every week, focus on post-earnings situations where:

- Earnings surprise is positive.
- Revenue surprise is positive.
- Estimate revisions are positive.
- Guidance or forward estimates improve.
- Price reaction is strong but not overextended.
- Options-implied move and liquidity are acceptable.

This may produce fewer trades but higher expected alpha per trade.

### Pivot B: Catalyst-Filtered Growth Basket

Keep the weekly basket, but only allow candidates with recent positive catalysts:

- Positive estimate revisions.
- Recent earnings beat.
- Positive price-target revision.
- Insider cluster buy.
- Institutional accumulation.
- Strong relative strength with improving fundamentals.

This can reduce bad top-k picks.

### Pivot C: Defensive Avoidance Model

Instead of only trying to find winners, train a model to avoid severe underperformers:

- Estimate cuts.
- Negative surprise.
- High short interest without positive catalyst.
- Negative filing events.
- Insider selling clusters.
- Liquidity/volatility deterioration.

Use it as a hard exclusion or position-size haircut.

### Pivot D: Higher-Frequency Event/Risk Overlay

If options/news/intraday data is acquired, build a shorter-horizon overlay around events. This is a bigger architectural change and should happen only after structured daily event data is proven useful.

## Implementation Plan After a Data Source Is Chosen

### Phase 1: Integration Scaffold

Files likely needed:

- `scripts/download_<vendor>_data.py`
- `scripts/build_<vendor>_features.py`
- `scripts/audit_<vendor>_data.py`
- `tests/test_<vendor>_data.py`
- `tests/test_<vendor>_features.py`
- metadata under `/root/.qlib/qlib_data/us_data/metadata/`
- raw data under `/root/.qlib/<vendor>/raw/` or another explicit external-data directory

Rules:

- Never commit API keys.
- Store raw provider responses append-only when possible.
- Write a provenance JSON with endpoint, pull date, parameters, row counts, ticker counts, date ranges, and schema hash.
- Add synthetic tests before large pulls.

### Phase 2: Point-in-Time Audit

Checks:

- No feature date earlier than provider availability date.
- No actual earnings/surprise before report release.
- No estimate revisions before publication date.
- No SEC filing text before accepted datetime.
- No 13F/SF3A data before lagged availability date.
- Missing data is explicit and does not silently forward-fill too far.

### Phase 3: Feature Build and Coverage Report

Outputs:

- Feature bins dumped into qlib.
- Coverage table by date and universe.
- Missingness by feature.
- Cross-sectional distribution checks.
- Outlier checks.
- Correlation with existing features.
- Feature freshness report.

### Phase 4: Target Audit

Before retraining, audit target usefulness:

- Rank buckets by year.
- Top/bottom spread by year.
- Event-conditioned target behavior.
- Benchmark-excess behavior versus QQQ/SPY/IXIC.
- Separate 2022, 2023, 2024, 2025, and 2026 slices.

### Phase 5: Controlled Training Runs

Run a small fixed candidate grid:

- Current baseline.
- New data only diagnostic score model.
- LightGBM rank/regression with new features.
- Event target.
- Catalyst-filtered target.
- Defensive avoidance overlay.

Avoid broad hyperparameter sweeps until a clear signal exists.

### Phase 6: Release Validation

Use the existing release runner with:

- QQQ, SPY, IXIC baselines.
- Strict gate profile.
- Stress costs.
- Trial registry.
- Walk-forward manifest.
- Feature provenance requirements.

### Phase 7: Paper Trading Before Live

Even after passing historical gates:

- Run paper trading or shadow trade-plan generation.
- Track live slippage versus assumptions.
- Track data update failures.
- Track missing features and late vendor data.
- Compare live candidate ranking stability to backtest behavior.
- Retire the candidate if rolling excess deteriorates beyond the current release policy.

## Decision Rule for Buying Data

If choosing only one paid source first:

- Choose analyst estimates plus earnings calendar.
- Prefer a vendor that gives historical point-in-time estimate snapshots, not just latest estimates.
- If FMP or EODHD cannot provide true PIT historical revisions, they may still help for earnings calendar and basic surprise data, but we should not rely on them as the final expectations source.
- If budget is high enough, get a quote from Intrinio/Zacks or another institutional estimates provider.

If choosing no paid source yet:

- Build SF2 insider features.
- Expand SF3A features.
- Add free SEC EDGAR filing-event metadata.
- Add free/cheap macro regime data.
- Then rerun the strict validation harness.

## Bottom Line

The current pipeline is promising as a research and validation harness, but the current Sharadar-only candidate models are not release-grade. The main problem is not that the code cannot train or validate a model. The main problem is that the current feature set does not consistently identify a profitable top basket across regimes.

The most useful data search should focus on forward-looking, timestamped, point-in-time information:

- analyst estimate revisions,
- earnings calendar and surprises,
- ratings and price-target changes,
- insider and institutional activity,
- filing events,
- short/crowding pressure,
- options-implied event risk.

The first paid pilot should be expectations plus earnings data. The first no-extra-cost implementation should be better SF2/SF3A features and SEC filing-event metadata.
