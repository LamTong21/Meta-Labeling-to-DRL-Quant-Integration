# Architectural Evolution of Multi-Asset Quantitative Trading Frameworks in Emerging Markets: From Statistical Meta-Labeling to Deep Reinforcement Learning

## Abstract

Designing algorithmic trading and portfolio allocation architectures for emerging equity markets presents severe challenges stemming from high noise-to-signal ratios, structural breaks, liquidity clustering, and transaction frictions. This study documents the theoretical rationale, mathematical formulation, empirical behavior, and systemic limitations observed during the architectural evolution between two consecutive iterations of an end-to-end quantitative framework (designated **Modeling09** and **Modeling10**) applied to the Vietnamese equity market (VN-Index universe).

**Modeling09** deployed a supervised, two-stage predictive architecture featuring geometric bar integrity auditing, econometric and signal-processing feature filtering, and triple-barrier meta-labeling optimized via Bayesian hyperparameter tuning. However, empirical verification reveals an inherent structural failure: optimizing symmetric statistical loss functions on directional cross-entropy fails to translate into convex portfolio utility under dynamic slippage and inventory constraints.

To resolve this divergence, **Modeling10** reformulated the allocation task into a continuous Markov Decision Process (MDP), substituting the isolated predictive and sizing blocks with an on-policy Deep Reinforcement Learning (DRL) agent operating under Proximal Policy Optimization (PPO) within a vectorized, multi-asset Gymnasium simulation.

We present the complete mathematical formalization of both paradigms, analyze their empirical diagnostic properties across 49 liquid equity assets from July 2018 through May 2026, and critically examine open operational pathologies, including policy collapse, non-stationarity drift, and cross-asset execution delays.

## 1. Introduction and Empirical Context

The efficient allocation of capital across risky assets requires solving two coupled problems:

1. **Signal extraction**: Isolating genuine informational alpha from non-stationary, heavy-tailed financial time series.
2. **Policy mapping**: Mapping noisy informational signals into an optimal sequence of continuous asset weights $w_t \in \mathbb{R}^N$ subject to non-negative cash holdings, turnover budgets, exchange price collars, and asymmetric transaction costs.

In developed markets, quantitative research frequently separates alpha generation from portfolio execution via mean-variance optimization (MVO), Black-Litterman shrinkage, or risk-parity engines. However, in an emerging market such as the Ho Chi Minh Stock Exchange (HOSE), this decoupled paradigm experiences structural failure due to distinctive microstructural frictions:

- **Statutory asymmetric volatility bands**: Daily price variation is strictly clamped (e.g., $\pm 7\%$ on HOSE), inducing artificial volatility suppression and truncated log-return tails.
- **Settlement latency and asymmetric liquidity**: Trading cycles (e.g., $T+2$ or $T+1.5$) limit intra-week hedging fluidity, creating liquidity lock-ups during severe market regimes.
- **Microstructure data corruption**: Unadjusted splits, dividend gaps, and prolonged trading suspensions distort historical bar representations.

The research codebases **Modeling09** and **Modeling10** capture a fundamental shift in quantitative methodology:

- **Modeling09** sought to tame these anomalies by combining data auditing with a supervised meta-labeling pipeline (combining econometric filtering, tree-based boosting via XGBoost/LightGBM, and probability calibration).
- **Modeling10** recognized the limitation of intermediate classification targets and unified signal generation, capital preservation, and turnover penalization directly into a sequential decision-making agent via Actor-Critic Deep Reinforcement Learning.

This paper details the mathematical underpinnings, empirical validation, architectural progression, and unresolved pathologies of this evolutionary milestone.

## 2. Fundamental Data Architecture and Microstructural Auditing

Both versions share a unified data engine instantiated via `MultiAssetDataPreparer`, designed to audit bar geometry and prevent look-ahead bias across $N = 49$ constituents and the benchmark index.

```
                     +---------------------------------------+
                     |         Raw Multi-Source Data         |
                     |  - Equities: yfinance (Daily OHLCV)   |
                     |  - Benchmark: vnstock Quote API (VCI) |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |     Geometric Candlestick Audit       |
                     |   Ensure Low <= min(O,C) &            |
                     |          High >= max(O,C)             |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Corporate Action Factor Normalization|
                     |     k_t = P_{t}^{close, adj} / P_t    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Exchange Calendar Alignment & Filter |
                     |   - Mask Trading Gaps (Delta t > 7)   |
                     |   - Left-Join Market Factor (VNINDEX) |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | Synchronized Tensor Pipeline:         |
                     | r_t, r_t^{overnight}, r_t^{intraday}, |
                     | sigma_t^{Parkinson}, r_t^{mkt}        |
                     +---------------------------------------+
```

### 2.1. Candlestick Geometry Verification

Public price feeds frequently suffer from transmission misalignments where high or low prices violate logical price boundaries. Let $O_{i,t}, H_{i,t}, L_{i,t}, C_{i,t}$ denote the raw Open, High, Low, and Close quotes for asset $i$ on day $t$. The pipeline enforces rigorous candlestick consistency:

$$
\tilde{H}_{i,t} = \max\left(H_{i,t}, O_{i,t}, C_{i,t}\right)
$$

$$
\tilde{L}_{i,t} = \min\left(L_{i,t}, O_{i,t}, C_{i,t}\right)
$$

subject to the physical domain constraint:

$$
\mathcal{D}_{\text{bar}} = \left\{ (O, H, L, C, V) \in \mathbb{R}_+^5 \;\middle\vert{}\; \tilde{L}_{i,t} > 0, \; V_{i,t} \ge 0 \right\}
$$

### 2.2. Corporate Action Proportionality

To eliminate phantom return spikes caused by stock dividends, bonus share issuances, and cash distributions, split adjustments are normalized via the ratio of dividend-adjusted close $C_{i,t}^{\text{adj}}$ to unadjusted close $C_{i,t}$:

$$
\kappa_{i,t} = \frac{C_{i,t}^{\text{adj}}}{C_{i,t}}
$$

$$
O_{i,t}^{\text{adj}} = O_{i,t} \cdot \kappa_{i,t}, \quad H_{i,t}^{\text{adj}} = \tilde{H}_{i,t} \cdot \kappa_{i,t}, \quad L_{i,t}^{\text{adj}} = \tilde{L}_{i,t} \cdot \kappa_{i,t}
$$

### 2.3. Return Decomposition and Trading Calendar Gap Masking

Asset price dynamics are decomposed into continuous compounding components:

$$
r_{i,t} = \ln\left(\frac{C_{i,t}^{\text{adj}}}{C_{i,t-1}^{\text{adj}}}\right)
$$

$$
r_{i,t}^{\text{overnight}} = \ln\left(\frac{O_{i,t}^{\text{adj}}}{C_{i,t-1}^{\text{adj}}}\right), \quad r_{i,t}^{\text{intraday}} = \ln\left(\frac{C_{i,t}^{\text{adj}}}{O_{i,t}^{\text{adj}}}\right)
$$

Extremum-based volatility proxy is calculated using the Parkinson log-range formulation:

$$
\sigma_{i,t}^{\text{Parkinson}} = \sqrt{\frac{1}{4\ln 2} \left[\ln\left(\frac{H_{i,t}^{\text{adj}}}{L_{i,t}^{\text{adj}}}\right)\right]^2}
$$

To prevent anomalous return inflation caused by regulatory delisting or prolonged administrative trading halts, we define an indicator variable over the calendar interval $\Delta \tau_t = \text{Date}_t - \text{Date}_{t-1}$:

$$
r_{i,t} := \text{NaN}, \quad r_{i,t}^{\text{overnight}} := \text{NaN} \quad \forall \; \Delta \tau_t > 7 \text{ calendar days}
$$

Exogenous market dynamics $r_{m,t} = \ln(C_{m,t}^{\text{mkt}} / C_{m,t-1}^{\text{mkt}})$ derived from the VN-Index benchmark are aligned with the asset time-index using a forward-fill left-join operator, imputing zero return for market holidays: $r_{m,t} \leftarrow 0.0$ if missing.

## 3. Modeling09: The Statistical Meta-Labeling Framework

### 3.1. Mathematical Architecture

**Modeling09** operationalized Marcos López de Prado’s meta-labeling philosophy to decouple the prediction of **direction** from the estimation of **bet sizing / execution conviction**.

```
+---------------------------------------------------------------------------------------+
|                               MODELING09 ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------+
|  Primary Model (Exogenous / Directional Signal)                                       |
|    - Signal Generator: f(X_t) -> y_hat_t in {-1, 0, 1}                                |
+---------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|  Triple-Barrier Event Labeling                                                        |
|    - Upper Barrier: +k * sigma_t                                                      |
|    - Lower Barrier: -k * sigma_t                                                      |
|    - Vertical Barrier: t + h                                                          |
|    - Target Meta-Label: y_t* = 1 if Upper touched first; else 0                       |
+---------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|  Multi-Stage Econometric & Signal-Processing Filtering                                |
|    - Stationarity: ADF, KPSS, BDS tests                                               |
|    - Serial Correlation & Volatility Clustering: Ljung-Box, ARCH-LM                   |
|    - Non-linear Dependence: Mutual Information I(X_j; Y*)                             |
|    - Feature Dimensionality Reduction & Regime Switching (Markov Autoregressive)      |
+---------------------------------------------------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|  Meta-Model Training & Probability Calibration                                        |
|    - Estimators: XGBoost, LightGBM, Random Forest                                     |
|    - Objective: min_theta Loss_binary(y_t*, P(y_t*=1 | X_t))                          |
|    - Calibration: Isotonic Regression / Platt Sigmoid Scaling                         |
|    - Execution Conviction: Bet Size w_t = 2 * (P_calibrated - 0.5)                     |
+---------------------------------------------------------------------------------------+
```

1. **Primary Directional Hypothesis**:
    
    A baseline econometric or moving-average crossover model generates a directional proposition $f(X_t) \to \hat{y}_t \in \{-1, 1\}$.
    
2. **Triple-Barrier Event Sampling**:
    
    A path-dependent framework defines the realization of an informational event over an investment horizon $h$. Let $\tau$ be the stopping time:
    
    $$
    \tau = \inf \left\{ s \in (t, t+h] \;\middle\vert{}\; \left\vert{}\ln\left(\frac{P_s}{P_t}\right)\right\vert{} \ge k \cdot \sigma_t \right\} \wedge (t+h)
    $$
    
    The meta-label $y_t^* \in \{0, 1\}$ is assigned:
    
    $$
    y_t^* = \begin{cases}     1, & \text{if } \frac{P_\tau}{P_t} \cdot \hat{y}_t > 1 + k \cdot \sigma_t \quad (\text{Signal confirmed with sufficient margin}) \\     0, & \text{otherwise} \quad (\text{Stop-loss hit or horizontal barrier reached without profit})     \end{cases}
    $$
    
3. **Multi-Stage Statistical Feature Filtering**:
Before feeding features $X_t$ into gradient boosting classifiers, features undergo sequential econometric verification:
    - *Stationarity*: Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test rejection bounds:
        
        $$
        \Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{p=1}^P \delta_p \Delta X_{t-p} + \varepsilon_t, \quad H_0: \gamma = 0
        $$
        
    - *Serial Correlation & ARCH effects*: Ljung-Box test on standardized residuals and Engle's ARCH-LM test on squared residuals:
        
        $$
        Q_{\text{LB}} = T(T+2)\sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m)
        $$
        
        $$
        \hat{\varepsilon}_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \hat{\varepsilon}_{t-i}^2 + \nu_t, \quad H_0: \alpha_1 = \dots = \alpha_q = 0
        $$
        
    - *Information-Theoretic Relevance*: Features are filtered based on non-linear Shannon entropy and mutual information criteria with the meta-target:
        
        $$
        I(X_j; Y^*) = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} \, dx \, dy \ge \xi
        $$
        
4. **Probability Calibration for Bet Sizing**:
Predictions generated by tree ensembles ($\hat{p}_t = \text{GBDT}(X_t)$) are uncalibrated under class-imbalanced financial regimes. To prevent overconfident capital allocation, **Modeling09** applies isotonic non-parametric regression:
    
    $$
    \hat{p}_{\text{cal}} = \arg\min_{m} \sum_{i=1}^M \left( y_i^* - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(a) \le m(b) \; \forall \; a \le b
    $$
    
    The targeted bet size corresponds to:
    
    $$
    w_t = \text{sign}(\hat{y}_t) \cdot g\left(\hat{p}_{\text{cal}, t}\right), \quad g(p) = \max\left(0, 2(p - 0.5)\right)
    $$
    

### 3.2. Structural Pathologies of Modeling09

Despite its theoretical elegance, **Modeling09** suffers from three severe flaws when deployed in live market simulations:

1. **Divergence of Classification Loss and Economic Utility**: The binary log-loss optimization criterion:
    
    $$
    \mathcal{L}_{\text{BCE}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \left[ y_t^* \ln \hat{p}_t + (1 - y_t^*) \ln (1 - \hat{p}_t) \right]
    $$
    
    is completely agnostic to market volatility regimes, skewness, and drawdown severity. A false negative during an explosive $15\%$ rally carries the same mathematical penalty as a false positive entering a cascading limit-down selloff.
    
2. **Static Inventory and Cash-Drag Ignorance**: The meta-labeling formulation assumes that each bet is independent. It cannot dynamically manage cash drag, portfolio turnover limits, or current inventory risk.
3. **Execution Delay in Sequential Regimes**: The model fails to capture auto-correlated price impacts where aggressive rebalancing in illiquid tickers triggers significant slippage.

## 4. Modeling10: The Deep Reinforcement Learning (DRL) Paradigm

### 4.1. Mathematical Formulation as a Markov Decision Process (MDP)

To resolve the limitations of Modeling09, **Modeling10** reformulates the continuous portfolio management task as an infinite-horizon discounted MDP represented by the 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

```
+-----------------------------------------------------------------------------------------+
|                                MODELING10 ARCHITECTURE                                  |
+-----------------------------------------------------------------------------------------+
|  Vectorized Multi-Asset Environment (Gymnasium Interface)                               |
|                                                                                         |
|  State Space s_t in S:                                                                  |
|    - Asset Return Vectors: r_{i, t-k}                                                   |
|    - Normalized Volume Dynamics: V_{i, t} / MA(V)_{i, 20}                               |
|    - Volatility Proxies: sigma_t^{Parkinson}                                            |
|    - Macro Factor / Market Regime: r_{m, t} (VN-INDEX)                                  |
|    - Current Portfolio Allocations: w_{t-1} in Delta^{N+1}                              |
+-----------------------------------------------------------------------------------------+
                                 |                          ^
                     Action a_t  |                          | Transition P(s_{t+1}|s_t, a_t)
                     (Weights)   v                          | Feedback Reward R_t
+-----------------------------------------------------------------------------------------+
|  Actor-Critic Deep Policy Network (PPO - Stable-Baselines3)                             |
|                                                                                         |
|  Actor Network pi_theta(a_t | s_t):                                                     |
|    - Parametrizes Dirichlet / Gaussian Policy -> Softmax Layer -> a_t in Delta^{N+1}    |
|    - Clipped Surrogate Objective:                                                       |
|        L^CLIP(theta) = E_t [ min( r_t(theta) A_hat_t, clip(r_t, 1-eps, 1+eps) A_hat_t )]|
|                                                                                         |
|  Critic Network V_phi(s_t):                                                             |
|    - Parametrizes Expected Value Baseline -> MSE Objective:                             |
|        L^VF(phi) = E_t [ ( V_phi(s_t) - G_t )^2 ]                                       |
|                                                                                         |
|  Integrated Reward Function R_t:                                                        |
|    R_t = ln( 1 + w_t^T r_t - Cost_turnover(w_t, w_{t-1}) ) - lambda * Variance_penalty  |
+-----------------------------------------------------------------------------------------+
```

**State Space ($\mathcal{S}$)**

The state vector $s_t \in \mathcal{S} \subset \mathbb{R}^{D}$ captures localized asset trajectories, cross-asset dependencies, macro regime indicators, and current portfolio positions:

$$
s_t = \left[ \mathbf{R}_{t-L:t}, \; \mathbf{\Sigma}_{t-L:t}^{\text{Parkinson}}, \; \mathbf{\tilde{V}}_{t-L:t}, \; r_{m, t-L:t}, \; \mathbf{w}_{t-1} \right]
$$

where:

- $\mathbf{R}_{t-L:t} \in \mathbb{R}^{N \times L}$ represents historical log returns across a lookback window $L$.
- $\mathbf{\Sigma}_{t-L:t}^{\text{Parkinson}} \in \mathbb{R}^{N \times L}$ captures short-term intraday volatility.
- $\mathbf{\tilde{V}}_{t-L:t} = V_{i, t} / \text{SMA}(V_{i}, 20)$ represents standardized trading volume ratios.
- $r_{m, t}$ denotes benchmark return dynamics.
- $\mathbf{w}_{t-1} \in \Delta^{N+1}$ represents the portfolio allocation vector inherited from the previous step.

**Action Space ($\mathcal{A}$)**

The action $a_t \in \mathcal{A} \subset \mathbb{R}^{N+1}$ defines target portfolio weights across $N$ risky assets and one risk-free cash component ($w_{0,t}$):

$$
\mathcal{A} = \left\{ \mathbf{w}_t \in \mathbb{R}^{N+1} \;\middle\vert{}\; \sum_{i=0}^N w_{i,t} = 1, \quad w_{i,t} \ge 0 \; \forall \; i \in \{0, \dots, N\} \right\}
$$

The action simplex $\Delta^{N+1}$ is guaranteed by applying a Softmax projection operator at the terminal layer of the policy network:

$$
w_{i,t} = \frac{\exp(z_{i,t})}{\sum_{j=0}^N \exp(z_{j,t})}
$$

**Transition Dynamics ($\mathcal{P}$)**

The state evolves based on market price realization. Portfolio equity develops under endogenous rebalancing costs:

$$
\mathbf{w}_{t}' = \frac{\mathbf{w}_t \odot (1 + \mathbf{r}_{t+1})}{\mathbf{w}_t^T (1 + \mathbf{r}_{t+1})}
$$

where $\mathbf{w}_{t}'$ represents the weight drift prior to active rebalancing at $t+1$.

**Reward Function ($\mathcal{R}$)**

Rather than minimizing an isolated classification or forecasting error, the reward directly incentivizes risk-adjusted portfolio growth while heavily penalizing turnover friction and extreme tail risks:

$$
R_t = \ln\left(1 + \mathbf{w}_t^T \mathbf{r}_t - c_{\text{trans}} \sum_{i=1}^N \vert{}w_{i,t} - w_{i,t-1}'\vert{}\right) - \lambda \cdot \left(\min\left(0, \mathbf{w}_t^T \mathbf{r}_t - \text{VaR}_{\alpha}\right)\right)^2
$$

where $c_{\text{trans}}$ denotes the variable transaction cost (brokerage fee + exchange duty + liquidity spread), and $\lambda$ penalizes downside tail risk (downside semi-variance).

### 4.2. Proximal Policy Optimization (PPO) Dynamics

**Modeling10** deploys an on-policy Actor-Critic network optimized via PPO (utilizing Gymnasium and Stable-Baselines3). The policy parameter vector $\theta$ is updated by maximizing the clipped surrogate objective function:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \; \text{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t \right) \right]
$$

where the probability ratio is defined as:

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

The advantage estimator $\hat{A}_t$ is computed using Generalized Advantage Estimation (GAE) parameterized by $\lambda_{\text{GAE}}$ and discount factor $\gamma$:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda_{\text{GAE}})^l \delta_{t+l}^V
$$

$$
\delta_t^V = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Simultaneously, the Critic network parameterized by $\phi$ minimizes the value estimation error:

$$
L^{\text{VF}}(\phi) = \hat{\mathbb{E}}_t \left[ \left( V_\phi(s_t) - \hat{G}_t \right)^2 \right], \quad \hat{G}_t = \hat{A}_t + V_\phi(s_t)
$$

## 5. Empirical Diagnostics and Execution Results

### 5.1. Dataset Profile and Physical Audit Verification

Empirical validation was performed on the liquid constituent universe of the Vietnamese equity market using historical bars covering July 2, 2018, to May 29, 2026. The target universe consists of $N = 49$ equities spanning blue-chip industrials, utilities, technology, and consumer staples (e.g., FPT, VNM, MWG, REE, BMP, DGC) alongside the VN-Index benchmark.

Execution of `raw_data_profile()` across both iterations confirmed the integrity of the data cleaning pipeline:

```
================================================================================
SAMPLE DATA AUDIT PROFILE (REPRESENTATIVE TICKERS):
- Time Span: 2018-07-02 to 2026-05-29 (2,052 to 2,061 trading sessions)
- Total Missing Values (NaN): 2 (Strictly isolated to t=0)
- Physical Bar Geometry: Min(High - max(Open, Close)) >= 0, Min(min(Open, Close) - Low) >= 0
================================================================================
```

```
Observed Empirical Missing Value Ledger (Across Universe Constituents):
+--------+------------------+---------------+-------------------------------------+
| Ticker | Session Count    | Total NaNs    | Localized Features of Missingness   |
+--------+------------------+---------------+-------------------------------------+
| ABT    | 2,056            | 2             | log_return (1), overnight_return (1)|
| ADP    | 733 (Late IPO)   | 2             | log_return (1), overnight_return (1)|
| BMP    | 2,052            | 2             | log_return (1), overnight_return (1)|
| BRC    | 2,052            | 2             | log_return (1), overnight_return (1)|
| CHP    | 2,061            | 2             | log_return (1), overnight_return (1)|
| CSM    | 2,052            | 2             | log_return (1), overnight_return (1)|
| CSV    | 2,052            | 2             | log_return (1), overnight_return (1)|
| CTI    | 2,061            | 2             | log_return (1), overnight_return (1)|
+--------+------------------+---------------+-------------------------------------+
```

*Empirical Confirmation of Lag Formulation*: For every asset evaluated, exactly **two** NaN values are observed over the 8-year span. These are strictly confined to index $t=0$ for $r_t = \ln(C_t / C_{t-1})$ and $r_t^{\text{overnight}} = \ln(O_t / C_{t-1})$, confirming the absence of unhandled missing data, unindexed weekends, or structural database drops.

### 5.2. Comparative Paradigm Analysis

The table below summarizes the quantitative, architectural, and operational characteristics of **Modeling09** versus **Modeling10**:

| **Architectural Attribute** | **Modeling09 (Supervised Meta-Labeling)  IPYNB** | **Modeling10 (Continuous Actor-Critic DRL)  IPYNB** |
| --- | --- | --- |
| **Core Algorithmic Engine** | LightGBM, XGBoost, Random Forest, Logistic Regression | Proximal Policy Optimization (PPO) via Stable-Baselines3 |
| **Problem Formulation** | Static pattern recognition with probability calibration | Infinite-horizon discounted Markov Decision Process (MDP) |
| **Objective Function** | Binary Cross-Entropy on Triple-Barrier Event: $\min \mathcal{L}_{\text{BCE}}$ | Clipped Generalized Advantage Objective: $\max L^{\text{CLIP}}(\theta)$ |
| **Target Representation** | Discrete ternary/binary label: $y_t^* \in \{0, 1\}$ | Continuous simplex weight action: $\mathbf{w}_t \in \Delta^{N+1}$ |
| **Turnover & Cost Awareness** | Ex-post penalty heuristic or static hurdle subtraction | Endogenous transaction penalty in step reward: $R_t \propto -c_{\text{trans}} \Vert{}\Delta \mathbf{w}_t\Vert{}_1$ |
| **Feature Selection Pipeline** | Multi-stage statistical filters (ADF, KPSS, ARCH, LB, MI) | Representation learning via Deep Multilayer Perceptrons |
| **Hyperparameter Tuning** | Bayesian Tree-structured Parzen Estimator (Optuna) on Trees | Hyperparameter schedules on Actor-Critic entropy and learning rates |
| **Execution Environment** | Scikit-learn tabular cross-validation (TimeSeriesSplit) | Vectorized multi-asset Gymnasium simulation (`DummyVecEnv`) |

## 6. Open Pathologies, Research Gaps, and Failure Modes

While **Modeling10** successfully addresses the objective misalignment of **Modeling09**, empirical testing reveals several critical vulnerabilities that remain open research problems in quantitative reinforcement learning.

```
+-----------------------------------------------------------------------------------------+
|                                    SYSTEMIC VULNERABILITIES                            |
+-----------------------------------------------------------------------------------------+
|  1. Action Simplex Saturation (Policy Collapse)                                         |
|     - High market volatility triggers risk-off behavior                                 |
|     - Policy collapses to w_{cash} -> 1.0 (Cash Absorbing State)                        |
+-----------------------------------------------------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|  2. Non-Stationary State Space Drift                                                    |
|     - Covariance matrix non-stationarity: Cov_t(r_i, r_j) != Cov_{t+k}(r_i, r_j)        |
|     - Fixed-window lookback fails during structural macro breaks                        |
+-----------------------------------------------------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|  3. Reward Function Asymmetry and Credit Assignment Failure                             |
|     - Asymmetric market impacts (drawdowns occur faster than recoveries)                |
|     - Sparse reward feedback obscures the causal impact of intermediate actions         |
+-----------------------------------------------------------------------------------------+
```

### 6.1. Action Simplex Saturation and Policy Collapse (The Cash-Trap Pathology)

Under the Softmax parameterization in environments with high variance and variable transaction fees, the Actor network frequently experiences policy collapse:

- During historical regimes featuring elevated drawdown risk (such as the 2022 market downturn), the policy identifies holding cash ($w_{0} \to 1.0$) as the minimum-variance action that avoids both turnover penalties and negative return shocks.
- Consequently, the entropy of policy distribution $\mathcal{H}(\pi_\theta(\cdot \mid s_t))$ drops to zero:
    
    $$
    \lim_{t \to \infty} \mathcal{H}(\pi_\theta) = -\sum_{i=0}^N w_i \ln w_i \to 0
    $$
    

Once trapped in this absorbing state, gradient updates through the clipped surrogate objective become vanishingly small ($\nabla_\theta L^{\text{CLIP}} \to 0$), rendering the agent unable to re-allocate into recovering assets.

### 6.2. Non-Stationary Covariance and Dimension Scalability

In **Modeling10**, concatenating raw historical feature tensors across $N = 49$ assets yields an observation dimension $D \approx 49 \times L \times K$. In an emerging market setting:

- The asset-to-asset empirical covariance matrix $\mathbf{\Sigma}_t \in \mathbb{R}^{N \times N}$ is structurally non-stationary:
    
    $$
    \mathbb{E}[\mathbf{\Sigma}_t] \neq \mathbb{E}[\mathbf{\Sigma}_{t+k}]
    $$
    
- As a result, fully connected policy layers overfit to transient cross-asset correlation regimes that dissipate out-of-sample. To mitigate this, future iterations require relational inductive biases, such as **Graph Attention Networks (GATs)** or **Spatial-Temporal Transformers**, where asset nodes dynamically attend to graph-structured industry classifications.

### 6.3. Delayed Credit Assignment under Asymmetric Vietnam Clearing (T+2 Latency)

A core limitation of the standard Gymnasium formulation in **Modeling10** is the assumption of immediate execution: action $a_t$ executed at Close $t$ generates active returns over the interval $(t, t+1]$.

- Under actual HOSE market microstructural mechanics, long equity positions are subject to $T+1.5$ settlement latency before secondary disposal is legally permitted.
- If the DRL policy allocates to an asset that enters a liquidity freeze (accumulating consecutive floor-limit declines), the actual state transition departs radically from the simulated transition probability $\mathcal{P}(s_{t+1} \mid s_t, a_t)$. Incorporating inventory-lock constraint masks directly into the action space represents an unresolved engineering hurdle.

## 7. Conclusion and Research Trajectory

The architectural progression from **Modeling09** to **Modeling10** demonstrates a decisive paradigm shift in applied quantitative finance:

- **Modeling09** established an audited, robust data foundation and applied rigorous statistical filtering and probability calibration to predict localized path-dependent events. However, it failed to bridge the gap between statistical accuracy and holistic portfolio utility.
- **Modeling10** reformulated the portfolio problem as a sequential decision process, resolving the intermediate objective mismatch by learning an optimal allocation policy directly through PPO.

Nevertheless, DRL models introduce new failure modes, including policy collapse into cash positions, sensitivity to non-stationary feature spaces, and settlement latency distortions.

Future work will focus on:

1. Replacing unconstrained Softmax policy heads with **regularized Dirichlet policies** to explicitly manage exploration entropy.
2. Integrating **Graph Neural Networks (GNNs)** into the policy backbone to model inter-asset market structures.
3. Incorporating microstructural settlement friction masks into the Gymnasium environment to reflect physical trading realities in emerging markets.

## References:

1. **Modeling09(Updated_Meta_Labeling_Framework).ipynb**. Internal Technical Codebase: Multi-Asset Geometric Audit, Econometric Signal Diagnostics, and Meta-Labeling Model Pipeline (2026).
2. **Modeling10(Freezed_Meta_Labeling_Framework).ipynb**. Internal Technical Codebase: Continuous Multi-Asset Portfolio Environment, Actor-Critic Engine, and Vectorized PPO Framework (2026).
3. López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
5. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 61-65.
6. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 987-1007.