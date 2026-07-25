# TradeBeginner RL Framework Spec

## 1) Overview

This document defines the future RL architecture for optimizing `Agent3Strategy` in TradeBeginner.
The current online system uses API-hosted LLM inference. The RL phase introduces an Agent Lightning + VERL training loop while preserving agent business logic.

- **Target policy**: Agent 3 strategy generation behavior.
- **Reward source**: Agent 6 deterministic backtest reward.
- **Serving path after training**: OpenAI-compatible endpoint powered by Agent Lightning runtime.

## 2) Agent Lightning Mapping

### State

At the strategy call boundary, state includes:

- `MarketSnapshot` from Agent 2.
- `SentimentDigest` from Agent 1 (TrendRadar-aligned processing pipeline).
- Current `PortfolioState`.
- Episodic retrievals (similar scenarios and outcomes).
- Semantic memory retrievals (patterns, user preferences, templates).

### Calls

For each episode, calls are captured as `(meta, input, output)`:

- Agent 1 tool calls (ticker research/news enrichment).
- Agent 2 tool calls (ticker technical/screening).
- Memory retrieval calls (episodic/semantic).
- Final LLM strategy call.

### Execution

One execution is the full call sequence used to produce one `Strategy`.

### Reward

- `r_N`: terminal reward from Agent 6 (`RewardSignal.terminal_reward`).
- `r_1..r_{N-1}`: `0` initially.
- Future extension: intermediate rewards for useful research calls and calibrated confidence.

## 3) Training-Agent Disaggregation

### Agent Lightning Server

- Runs on GPU.
- Executes PPO/GRPO via VERL.
- Aggregates traces and rewards.
- Serves latest policy with OpenAI-compatible API.

### Agent Lightning Client

- Executes the same Agent 3 orchestration logic used online.
- Performs rollout collection.
- Computes reward locally using Agent 6 deterministic evaluator.
- Reports `(trace, reward)` to server.

This separation keeps training logic independent of business agent code.

## 4) RL Data Flow

1. **Rollout Generation**
   - Server dispatches historical scenario tasks.
   - Client builds context and runs Agent 3.
   - Agent 6 computes reward from historical market data.
   - Client returns trace + reward.

2. **Training Step**
   - Server batches traces.
   - VERL computes advantages and updates model.
   - New checkpoint deployed behind OpenAI-style API.

3. **Iteration**
   - Repeat over epochs.
   - Evaluate on holdout scenarios.

## 5) Dataset Construction

Task schema:

- `date`
- serialized market snapshot
- serialized sentiment digest
- serialized portfolio state
- labels for market regime
- enough future bars for backtest horizon

Recommended start:

- 1000+ tasks minimum.
- Regime-balanced slices: bull, bear, volatile, sideways.
- Split: 70/15/15 train/val/test.

## 6) Model Selection

Initial candidates:

- Qwen 2.5 (7B/14B),
- Llama 3 8B,
- DeepSeek variants.

Selection criteria:

- Reliable structured JSON output.
- Stable SFT/RL tooling support.
- Cost/performance ratio for iterative tuning.

## 7) Reward Shaping

Current reward composition:

- sharpe component,
- drawdown penalty component,
- win-rate component.

Future additions:

- research quality bonus,
- confidence calibration bonus,
- diversification stability term.

Stability rules:

- per-batch reward normalization,
- clipping to bounded range,
- low-noise deterministic backtesting to reduce reward variance.

## 8) Curriculum

Progressive training phases:

1. Easy trend regimes with strong directional signal.
2. Mixed regimes with moderate noise.
3. High-noise and low-signal regimes.

## 9) Evaluation Protocol

Compare:

- base model,
- RL-tuned model,
- simple baselines (buy-and-hold SPY, momentum/mean-reversion heuristics).

Metrics:

- mean reward,
- reward variance,
- sharpe, drawdown, win rate,
- JSON validity and schema adherence.

## 10) Infrastructure

Suggested baseline:

- 1x A100 80GB for 7B class.
- 50GB checkpoint storage budget.
- 10GB+ for replay/traces.

## 11) Migration Plan

1. Validate full online pipeline with API LLM.
2. Build historical scenario dataset.
3. Stand up Agent Lightning + VERL training server.
4. Point `llm_base_url` to server endpoint for dry-run inference.
5. Run RL training and holdout evaluation.
6. Promote tuned model if metrics improve.
7. Continue collecting behavior-cloning data from Agent 5.

