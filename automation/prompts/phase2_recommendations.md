# Phase 2: Recommendations & Improvement Ideas

You are running as a scheduled daily code review for the Backtester project. This is a **READ-ONLY** phase — do NOT modify any files, create branches, or make commits.

Read the codebase systematically to understand its current state, then produce a comprehensive recommendations report.

## Instructions

### 1. Architecture Assessment

Review the overall codebase structure and assess:

- **Coupling**: Are modules tightly coupled? Are there circular dependencies? Could components be swapped out more easily?
- **Config management**: Is configuration centralized? Are magic numbers scattered through the code? Are API keys handled safely?
- **Error handling patterns**: Is error handling consistent? Are failures logged with enough context to debug? Are retries implemented where needed?
- **Code duplication**: Are there repeated patterns that should be abstracted? Similar logic in multiple files?
- **Testing**: What's the test coverage? Are critical trading calculations tested? Are there integration tests?
- **Data integrity**: Are CSV read/writes robust? Could data corruption occur? Are there validation checks on data pipeline inputs/outputs?

### 2. Feature Opportunities

Think like a quantitative trader. What features would materially improve trading analysis and decision-making?

Consider these categories:

- **Risk management**: Position sizing optimization, portfolio-level risk limits, correlation-based exposure tracking, drawdown alerts, max loss circuit breakers
- **Pattern recognition**: New setup type detection, regime classification (trending/mean-reverting/choppy), sector rotation signals, time-of-day performance patterns
- **Execution quality**: Slippage analysis, fill rate tracking, timing analysis (how much edge is lost to late entries/exits), optimal order types
- **Backtesting improvements**: Walk-forward optimization, Monte Carlo simulation, out-of-sample validation, parameter sensitivity analysis, market regime-conditional performance
- **Data enrichment**: Options flow integration, short interest data, institutional flow signals, earnings/catalyst calendar awareness, sector/factor decomposition
- **Monitoring & alerting**: Real-time P&L tracking, strategy drift detection, anomaly alerts (unusual correlations, volume spikes), end-of-day automated reports

### 3. Technical Debt Inventory

Identify code that works but is fragile, hard to maintain, or will cause pain as the project grows:

- Hardcoded values that should be configurable
- Functions that do too many things
- Missing type hints on critical interfaces
- Inconsistent date/time handling
- Fragile file path assumptions
- Missing logging in critical paths

### 4. Quick Wins

Identify improvements that are:
- **Small**: Less than 1 hour of work
- **High impact**: Meaningfully improve reliability, usability, or performance
- **Low risk**: Unlikely to break existing functionality

Examples: adding a missing null check, centralizing a repeated constant, adding a retry to a flaky API call.

### 5. Output Report

Output your complete analysis as a structured markdown report:

```markdown
# Backtester Recommendations Report — {TODAY_DATE}

## Architecture Assessment

### Strengths
- ...

### Concerns
| # | Area | Description | Severity |
|---|------|-------------|----------|
| 1 | ... | ... | High/Medium/Low |

## Feature Opportunities

### Risk Management
- ...

### Pattern Recognition
- ...

### Execution Quality
- ...

### Backtesting Improvements
- ...

### Data Enrichment
- ...

### Monitoring & Alerting
- ...

## Technical Debt

| # | File(s) | Description | Impact | Effort |
|---|---------|-------------|--------|--------|
| 1 | ... | ... | High/Medium/Low | Hours/Days |

## Quick Wins

| # | Description | File(s) | Impact | Effort |
|---|-------------|---------|--------|--------|
| 1 | ... | ... | ... | ~Xmin |

## Top 5 Recommendations (Ranked by Impact-to-Effort)

| Rank | Recommendation | Category | Impact | Effort | Rationale |
|------|---------------|----------|--------|--------|-----------|
| 1 | ... | ... | High | Low | ... |
| 2 | ... | ... | High | Medium | ... |
| 3 | ... | ... | Medium | Low | ... |
| 4 | ... | ... | High | High | ... |
| 5 | ... | ... | Medium | Medium | ... |

## Summary

- Architecture health: X/10
- Biggest risk: ...
- Highest-value next step: ...
```
