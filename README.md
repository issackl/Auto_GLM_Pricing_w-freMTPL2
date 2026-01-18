# Auto_GLM_Pricing_w-freMTPL2

# Auto Insurance Pricing Study – Frequency–Severity GLM

## Overview
This project implements an end-to-end auto insurance pricing analysis using a frequency–severity framework. Claim frequency and claim severity are modeled separately using generalized linear models (GLMs), with results validated, interpreted, and implemented in Excel to support actuarial pricing decisions.

The objective is to estimate **indicated pure premium**, assess model calibration through diagnostic comparisons, and translate GLM outputs into a practical pricing and quoting tool.

---

## Dataset
This project uses the **freMTPL2** motor third-party liability insurance dataset, a publicly available dataset commonly used for actuarial pricing research and education.

**Dataset:** French Motor Third-Party Liability (freMTPL2)

**Data structure:**
- Policy-level exposure and claim counts
- Claim-level loss amounts
- Driver, vehicle, and rating characteristics

**Key variables used:**
- Exposure (policy-year basis)
- Claim count (`ClaimNb`)
- Claim amount (loss severity)
- Driver age
- Vehicle age
- Bonus–malus (experience rating factor)
- Vehicle characteristics

The separation of policy-level and claim-level data enables a proper frequency–severity modeling approach consistent with industry practice.

---

## Methodology

### Frequency Model
- **Target:** Claim count  
- **Distribution:** Poisson  
- **Offset:** Log of exposure  
- **Purpose:** Estimate expected claim frequency per policy-year  

### Severity Model
- **Target:** Average claim severity  
- **Distribution:** Gamma (log link)  
- **Weights:** Claim count  
- **Purpose:** Estimate expected loss per claim  

### Pure Premium
Pure premium is calculated as: Frequency × Severity


