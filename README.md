# PBL-4 — Distributed CPS Fault Prediction Framework

**Course:** Project Based Learning 4, VI Semester, Jan–May 2026
**Department:** Computer Science and Engineering, Manipal University Jaipur
**Guide:** Dr. Akshay Jadhav
**Student:** Aaditya Upadhyay (23FE10CSE00457)

## Project

**Title:** A Distributed Lightweight Framework for Robust, Energy-Efficient, and Early Fault Prediction in Cyber-Physical Systems
**Dataset:** NASA C-MAPSS FD001 — 100 turbofan engines, 14 sensors, 20,631 timesteps
**Paper Status:** In active development. Targeting IEEE ICPS 2026.

## Key Results

| Metric | Value |
|--------|-------|
| Best centralized F1 (validation) | 0.864 |
| Distributed F1 — test partition | 0.839 |
| Centralized F1 — test partition | 0.841 |
| Wilcoxon p-value | 0.524 |
| Distributed F1 at 30% node dropout | 0.856 |
| Centralized F1 at 30% node dropout | 0.000 |
| Mean detection delay | -2.5 cycles |
| FD003 cross-dataset F1 | 0.753 |

## Pipeline

Phase 1 through Phase 10 notebooks are in cps/notebooks/. Results in cps/results/.

## Dataset

NASA C-MAPSS FD001 from the NASA Prognostics Data Repository. Not included here per licensing terms.

## Requirements

pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, xgboost>=1.7, scipy>=1.11, matplotlib>=3.8, seaborn>=0.13
