# 03_Implementation — 30-Seed Statistical Benchmark & Local Reproduction

This module contains the multi-core Python implementation and 30-seed statistical evaluation pipeline for **"Evolution Strategies as a Scalable Alternative to Reinforcement Learning"** (Salimans et al., 2017).

---

## 1. Environment Setup & Dependencies

* **Python Version:** Python 3.10+
* **Core Libraries:** `numpy>=1.24.0`, `scipy>=1.10.0`, `gymnasium[mujoco]>=0.29.0`, `mujoco>=2.3.0`, `matplotlib>=3.7.0`, `imageio>=2.31.0`, `imageio-ffmpeg>=0.4.8`

```bash
cd 03_Implementation
pip install -r requirements.txt