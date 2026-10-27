# Optimizer Benchmark Report

**Date:** December 6, 2025  
**Test Configuration:** Degree 5, 25 control points, G1 continuity, Parametric error metric  
**Airfoils Tested:** 41 diverse geometries from UIUC database  
**Timeout:** 300 seconds per optimizer per airfoil

---

## Executive Summary

SLSQP achieves the highest accuracy (6.97×10⁻⁵ average error) but requires ~222 seconds per fit. BFGS delivers 90% of SLSQP's accuracy in **26× less time** (8.6 seconds), making it the recommended default for interactive use.

---

## Test Methodology

### Airfoil Selection (41 total)

| Category | Airfoils | Rationale |
| ---------- | ---------- | ----------- |
| **NACA 4-digit** | 0006, 0008, 0015, 0018, 1408, 1412, 2408, 2412, 2415, 4412, 4418, 6412 | Symmetric and cambered variants, varying thickness |
| **NACA 5-digit** | 23012 | Complex camber line |
| **Eppler** | E193, E205, E211, E374, E395, E423 | High-performance sailplane sections |
| **Selig-Donovan** | SD7003, SD7037, SD7062, SD7080, SD8020 | Low Reynolds number designs |
| **Selig** | S1210, S1221, S2027, S3010, S6063, S8036 | General aviation and wind turbine |
| **Martin Hepperle** | MH32, MH60, MH114 | Modern RC airfoils |
| **Specialty** | RG15, HQ3510, SA7035, RAE2822, Whitcomb, AH93W145, GOE795 | Varied design philosophies |

### Optimizers Tested

All 7 local optimizers from SciPy's `minimize` function:

- **Gradient-based:** BFGS, L-BFGS-B, CG, SLSQP
- **Derivative-free:** Nelder-Mead, Powell, COBYLA

---

## Results

### Overall Performance Summary

| Rank | Optimizer | Avg Error | Avg Time | Timeouts | Speed Factor |
| ------ | ----------- | ----------- | ---------- | ---------- | -------------- |
| 1 | **SLSQP** | 6.97×10⁻⁵ | 222.3s | 0/41 | 1.0× (baseline) |
| 2 | **BFGS** | 7.77×10⁻⁵ | 8.6s | 0/41 | **25.8×** |
| 3 | L-BFGS-B | 1.74×10⁻⁴ | 3.4s | 0/41 | 65.4× |
| 4 | CG | 1.83×10⁻⁴ | 10.5s | 0/41 | 21.2× |
| 5 | Powell | 2.19×10⁻⁴ | 171.1s | **28/41** | 1.3× |
| 6 | COBYLA | 2.83×10⁻³ | 10.0s | 0/41 | 22.2× |
| 7 | Nelder-Mead | 5.15×10⁻³ | 4.9s | 0/41 | 45.4× |

### Win Distribution (Best Error per Airfoil)

```text
SLSQP:    ████████████████████████████████████████████████████████████  30 wins (73%)
BFGS:     ████████████████████  10 wins (24%)
L-BFGS-B: ██  1 win (2%)
Others:   0 wins
```

### Head-to-Head: BFGS vs SLSQP

| Metric | BFGS | SLSQP | Ratio |
| -------- | ------ | ------- | ------- |
| Mean Error | 7.77×10⁻⁵ | 6.97×10⁻⁵ | 1.11× worse |
| Median Error | 6.82×10⁻⁵ | 6.70×10⁻⁵ | 1.02× worse |
| Mean Time | 8.6s | 222.3s | **25.8× faster** |
| Median Time | 9.2s | 223.6s | **24.3× faster** |
| Reliability | 41/41 | 41/41 | Equal |

**Key insight:** BFGS achieves 89% of SLSQP's accuracy while being 26× faster.

---

## Per-Airfoil Results

### SLSQP Wins (30 airfoils)

| Airfoil | SLSQP Error | BFGS Error | SLSQP Advantage |
| --------- | ------------- | ------------ | ----------------- |
| naca4418 | 2.40×10⁻⁶ | 1.78×10⁻⁵ | 7.4× better |
| naca0018 | 3.94×10⁻⁶ | 1.14×10⁻⁵ | 2.9× better |
| naca0015 | 4.81×10⁻⁶ | 9.68×10⁻⁶ | 2.0× better |
| naca2415 | 1.19×10⁻⁵ | 2.36×10⁻⁵ | 2.0× better |
| sd7062 | 1.79×10⁻⁵ | 3.88×10⁻⁵ | 2.2× better |
| ... | ... | ... | ... |

### BFGS Wins (10 airfoils)

| Airfoil | BFGS Error | SLSQP Error | BFGS Advantage |
| --------- | ------------ | ------------- | ---------------- |
| goe795 | 1.11×10⁻⁴ | 1.30×10⁻⁴ | 1.2× better |
| mh32 | 1.76×10⁻⁴ | 1.98×10⁻⁴ | 1.1× better |
| naca6412 | 5.48×10⁻⁵ | 6.45×10⁻⁵ | 1.2× better |
| naca23012 | 1.79×10⁻⁵ | 2.14×10⁻⁵ | 1.2× better |
| e211 | 7.87×10⁻⁵ | 8.43×10⁻⁵ | 1.1× better |
| sd7037 | 6.82×10⁻⁵ | 7.31×10⁻⁵ | 1.1× better |
| s2027 | 5.45×10⁻⁵ | 5.76×10⁻⁵ | 1.1× better |
| s8036 | 2.55×10⁻⁵ | 2.89×10⁻⁵ | 1.1× better |
| rae2822 | 6.66×10⁻⁵ | 6.96×10⁻⁵ | 1.0× better |
| whitcomb | 3.44×10⁻⁵ | 3.75×10⁻⁵ | 1.1× better |

**Pattern:** BFGS wins tend to be narrow margins (1.0-1.2×), suggesting it found equivalent local minima faster. SLSQP wins can be substantial (2-7×), indicating deeper optimization.

---

## Optimizer Characteristics

### Tier 1: Production Ready

#### BFGS

- **Strengths:** Excellent accuracy/speed tradeoff, 100% reliability, consistent performance
- **Weaknesses:** Slightly lower accuracy than SLSQP on most airfoils
- **Best for:** Interactive GUI, iterative design, batch processing
- **Recommendation:** **Default optimizer**

#### SLSQP

- **Strengths:** Highest accuracy, handles constraints natively, 100% reliability
- **Weaknesses:** 26× slower than BFGS
- **Best for:** Final publication fits, when accuracy is paramount
- **Recommendation:** **Secondary option for refinement**

### Tier 2: Acceptable Alternatives

#### L-BFGS-B

- **Strengths:** Fastest optimizer (3.4s), memory-efficient
- **Weaknesses:** 2.2× worse accuracy than BFGS
- **Best for:** Very large batch jobs where speed trumps accuracy

#### CG (Conjugate Gradient)

- **Strengths:** Reasonable accuracy, moderate speed
- **Weaknesses:** No clear advantage over BFGS
- **Best for:** Specific problems where CG converges better

### Tier 3: Not Recommended

#### Powell

- **Fatal flaw:** 68% timeout rate (28/41 airfoils exceeded 5 minutes)
- **Do not use** for production

#### Nelder-Mead

- **Fatal flaw:** 74× worse accuracy than SLSQP
- **Do not use** for spline fitting

#### COBYLA

- **Fatal flaw:** 41× worse accuracy than SLSQP
- **Do not use** for spline fitting

---

## Recommendations

### Default Configuration

```python
OptimizationConfig(
    optimizer=OptimizerType.BFGS,  # Best speed/accuracy tradeoff
    metric=ErrorMetricType.PARAMETRIC,
    max_iterations=1000,
)
```

### Use Case Matrix

| Scenario | Recommended Optimizer | Expected Time | Expected Error |
| ---------- | ---------------------- | --------------- | ---------------- |
| Interactive GUI editing | BFGS | ~9s | ~8×10⁻⁵ |
| Quick preview | L-BFGS-B | ~3s | ~2×10⁻⁴ |
| Publication-quality fit | SLSQP | ~220s | ~7×10⁻⁵ |
| Batch processing (100+ airfoils) | BFGS | ~15 min | ~8×10⁻⁵ |
| Maximum accuracy (time unlimited) | SLSQP | ~4 hours | ~7×10⁻⁵ |

### Implementation Suggestion

Consider a two-stage approach in the GUI:

1. **Fast preview:** Run BFGS for immediate feedback (~9s)
2. **Refine button:** Option to run SLSQP for maximum accuracy (~220s)

---

## Appendix: Raw Data

Full benchmark results are stored in:

- `optimizer_logs/20251206_040753_benchmark_summary.csv`
- Individual optimizer logs in `optimizer_logs/`

### Test Environment

- Python 3.12
- SciPy (minimize function)
- Hardware: [User's machine]
- OS: Windows

---

*Report generated from benchmark script: `scripts/benchmark_optimizers.py`*
