# Architectural Evolution of Multi-Asset Quantitative Trading Frameworks in Emerging Markets: From Statistical Meta-Labeling to Deep Reinforcement Learning

![Python 3.10+](https://www.python.org/)
![Code style: black](https://github.com/psf/black)
![License: MIT](LICENSE)

An end-to-end quantitative research framework documenting the paradigm transition from a supervised two-stage econometric meta-labeling pipeline (**Modeling09**) to an on-policy continuous Deep Reinforcement Learning (DRL) agent operating under Proximal Policy Optimization (**Modeling10**), customized for emerging equity market microstructures (VN-Index / HOSE).

---

## 1. Theoretical Architecture

Allocating capital dynamically across emerging equity assets requires simultaneously addressing two coupled problems:
1. **Signal Extraction:** Isolating genuine informational alpha from heavy-tailed, non-stationary market time series under regulatory price distortion.
2. **Policy Mapping:** Mapping noisy signals into an optimal sequence of continuous asset weights $w_t \in \Delta^{N+1}$ subject to inventory locks, statutory exchange bands ($\pm 7\%$), and asymmetric liquidity costs.

### Architectural Paradigm Comparison

| Feature / Dimension | Modeling09 (Supervised Meta-Labeling) | Modeling10 (Continuous Actor-Critic DRL) |
| :--- | :--- | :--- |
| **Algorithmic Engine** | LightGBM, XGBoost, Random Forest | Proximal Policy Optimization (PPO) |
| **Problem Formulation** | Static pattern recognition + Isotonic calibration | Infinite-horizon discounted MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ |
| **Objective Target** | Binary cross-entropy $\min \mathcal{L}_{\text{BCE}}$ on events | Clipped generalized advantage $\max L^{\text{CLIP}}(\theta)$ |
| **Output Space** | Discrete event confidence: $y_t^* \in \{0, 1\}$ | Continuous simplex weight action: $\mathbf{w}_t \in \Delta^{N+1}$ |
| **Turnover Awareness** | Ex-post hurdle subtraction / sizing heuristic | Endogenous step penalty: $R_t \propto -c_{\text{trans}}\Vert{}\Delta \mathbf{w}_t\Vert{}_1$ |
| **Downside Risk** | Symmetric loss (same penalty for rallies/crashes) | Direct downside semi-variance penalization $(\mathbf{w}^T\mathbf{r} - \text{VaR}_\alpha)^2$ |
| **Execution Environment** | Scikit-learn tabular `TimeSeriesSplit` | Vectorized multi-asset Gymnasium simulation |

---

## 2. Mathematical Formalization

### 2.1. Physical Candlestick Auditing and Volatility

Let $O_{i,t}, H_{i,t}, L_{i,t}, C_{i,t}$ denote raw quotes. The data engine strictly enforces candlestick consistency before calculating returns:
$$\tilde{H}_{i,t} = \max\left(H_{i,t}, O_{i,t}, C_{i,t}\right), \quad \tilde{L}_{i,t} = \min\left(L_{i,t}, O_{i,t}, C_{i,t}\right)$$

Split adjustments use corporate action factor ratio $\kappa_{i,t} = C_{i,t}^{\text{adj}} / C_{i,t}$. Extremum-based volatility is computed using the Parkinson log-range formulation:
$$\sigma_{i,t}^{\text{Parkinson}} = \sqrt{\frac{1}{4\ln 2} \left[\ln\left(\frac{H_{i,t}^{\text{adj}}}{L_{i,t}^{\text{adj}}}\right)\right]^2}$$

To prevent anomalous return inflation from administrative halts, returns across intervals where $\Delta \tau_t > 7$ calendar days are masked as `NaN`.

### 2.2. Modeling09: Triple-Barrier Meta-Labeling

Given a directional proposition $\hat{y}_t \in \{-1, 1\}$, stopping time $\tau$ over horizon $h$ is defined as:
$$\tau = \inf \left\{ s \in (t, t+h] \;\middle\vert\; \left\vert{}\ln\left(\frac{P_s}{P_t}\right)\right\vert{} \ge k \cdot \sigma_t \right\} \wedge (t+h)$$

The binary meta-label $y_t^* \in \{0, 1\}$ classifies whether the position exited profitably before hitting the stop loss or expiring:
$$y_t^* = \begin{cases} 1, & \text{if } \frac{P_\tau}{P_t} \cdot \hat{y}_t > 1 + k \cdot \sigma_t \\ 0, & \text{otherwise} \end{cases}$$

Raw gradient-boosted probabilities are calibrated via monotonic isotonic regression:
$$\hat{p}_{\text{cal}} = \arg\min_{m} \sum_{i=1}^M \left( y_i^* - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(a) \le m(b) \; \forall \; a \le b$$

Continuous conviction bet sizing is assigned as:
$$w_t = \text{sign}(\hat{y}_t) \cdot \max\left(0, 2(\hat{p}_{\text{cal}, t} - 0.5)\right)$$

### 2.3. Modeling10: Continuous MDP Formulation

* **State Space ($\mathcal{S}$):**
  $$s_t = \left[ \mathbf{R}_{t-L:t}, \; \mathbf{\Sigma}_{t-L:t}^{\text{Parkinson}}, \; \mathbf{\tilde{V}}_{t-L:t}, \; r_{m, t-L:t}, \; \mathbf{w}_{t-1} \right] \in \mathbb{R}^D$$
* **Action Space ($\mathcal{A}$):** Unconstrained logits projected via Softmax onto the unit probability simplex $\Delta^{N+1}$:
  $$w_{i,t} = \frac{\exp(z_{i,t})}{\sum_{j=0}^N \exp(z_{j,t})}, \quad \sum_{i=0}^N w_{i,t} = 1, \quad w_{i,t} \ge 0$$
* **Weight Drift Transition ($\mathcal{P}$):**
  $$\mathbf{w}_{t}' = \frac{\mathbf{w}_t \odot (1 + \mathbf{r}_{t+1})}{\mathbf{w}_t^T (1 + \mathbf{r}_{t+1})}$$
* **Economic Reward Function ($\mathcal{R}$):**
  $$R_t = \ln\left(1 + \mathbf{w}_t^T \mathbf{r}_t - c_{\text{trans}} \sum_{i=1}^N \vert{}w_{i,t} - w'_{i,t-1}\vert{}\right) - \lambda \left(\min\left(0, \mathbf{w}_t^T \mathbf{r}_t - \text{VaR}_{\alpha}\right)\right)^2$$
* **PPO Clipped Objective:**
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \; \text{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t \right) \right]$$

---

## 3. Microstructural Frictions (HOSE)

The simulation engine explicitly models constraints present on the Ho Chi Minh Stock Exchange:
* **Statutory Asymmetric Price Bands:** Daily equity log-returns are truncated within daily regulatory bounds:
  $$r_{i,t} \in \left[\ln(1 - 0.07), \ln(1 + 0.07)\right]$$
* **$T+1.5$ Settlement Inventory Lock:** Capital allocated to equity positions at trading day $t$ cannot be liquidated until secondary settlement clearing completes. The execution model enforces an internal queue that tracks locked equity slices, restricting sell-side actions from executing before delivery.

---

## 4. Empirical Missingness Ledger

Physical bar auditing across the 49 liquid equity universe from July 2018 through May 2026 validates data integrity:

```text
================================================================================
SAMPLE DATA AUDIT PROFILE:
- Time Span: 2018-07-02 to 2026-05-29 (2,052 to 2,061 trading sessions)
- Total Missing Values (NaN): Exactly 2 per ticker (Isolated strictly to t=0)
- Physical Bar Geometry: Min(High - max(Open, Close)) >= 0, Min(min(Open, Close) - Low) >= 0
================================================================================

Across all constituent tickers, exactly two `NaN` values occur, originating from the initial backward difference lag operator for $\ln(C_t / C_{t-1})$ and $\ln(O_t / C_{t-1})$ at index $t=0$.

## 5. Installation and Environment Setup

Clone the repository and install the production dependencies:

```
git clone https://github.com/your-username/emerging-market-quant-drl.git
cd emerging-market-quant-drl

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 6. Execution Pipeline (CLI)

### Step 1: Run Physical Candlestick Audit

Ingests OHLCV bars, applies physical geometry boundaries, calculates return components, and saves the missingness ledger:

```
python -m scripts.run_data_audit --portfolio data/raw/final_portfolio.csv --start 2018-07-01 --end 2026-06-01
```

### Step 2: Train & Evaluate Modeling09 (Meta-Labeling)

Runs econometric filters (ADF/KPSS), applies triple-barrier sampling, trains cross-validated LightGBM models, applies isotonic calibration, and backtests the strategy:

```
python -m scripts.run_meta_labeling --portfolio data/raw/final_portfolio.csv --estimator lightgbm --horizon 5 --pt-sl 1.5
```

### Step 3: Train Modeling10 (Deep Reinforcement Learning)

Initializes the multi-asset Gymnasium simulation environment and trains the PPO Actor-Critic policy:

```
python -m scripts.run_drl_training --portfolio data/raw/final_portfolio.csv --policy MlpPolicy --timesteps 150000 --ent-coef 0.01
```

### Step 4: Compare Paradigms & Check Systemic Pathologies

Generates comparative institutional performance tear sheets and tests for policy entropy collapse:

```
python -m scripts.compare_models
```

## 7. Systemic Vulnerabilities & Open Pathologies

1. **Policy Collapse (Cash-Trap Pathology):** Under high market turbulence, standard Softmax policy heads can identify cash ($w_0 \to 1.0$) as an absorbing state that avoids both volatility and turnover penalties. Policy entropy collapses:
    
    $$\lim_{t \to \infty} \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \to 0$$
    
    Once trapped, policy gradient updates diminish ($\nabla_\theta L^{\text{CLIP}} \to 0$), impairing re-allocation into recovering assets. The included `DirichletActorCriticPolicy` mitigates this failure mode.
    
2. **Non-Stationary Covariance Matrices:** Fixed observation lookback windows overfit to transient cross-asset correlation regimes in emerging markets, necessitating graph-attention or spatial-temporal inductive biases.
3. **Execution Lag Distortions:** Unmodelled clearing friction can decouple actual portfolio transitions from theoretical MDP simulations when limit-down liquidity lockups occur.