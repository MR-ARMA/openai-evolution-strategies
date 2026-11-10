# OpenAI Evolution Strategies (ES) — Experimental Reproduction Report

## 1. Final Performance Summary (Table 2 Comparison)

| Environment | Generations | Total Timesteps | Initial Return (Gen 1) | Final Best Return | Training Time (min) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Halfcheetah-v4** | 150 | 9,600,000 | -0.6 | **5059.3** | 3.0m |
| **Hopper-v4** | 150 | 7,900,120 | 112.8 | **2045.1** | 4.9m |
| **Swimmer-v4** | 150 | 9,600,000 | -24.7 | **359.6** | 3.7m |
| **Walker2d-v4** | 150 | 6,750,131 | 68.7 | **1919.5** | 4.6m |

## 2. Evolutionary Progression Across Generations

### Halfcheetah-v4 Milestone Trajectory

| Generation | Timesteps | Population Mean | Parent Eval Return | Best Return |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 64,000 | -3.2 | -0.6 | **-0.6** |
| Gen 0038 | 2,432,000 | 2023.7 | 2457.0 | **2457.0** |
| Gen 0075 | 4,800,000 | 2716.4 | 3402.0 | **3402.0** |
| Gen 0112 | 7,168,000 | 3402.6 | 4187.7 | **4464.2** |
| Gen 0150 | 9,600,000 | 4110.5 | 5059.3 | **5059.3** |

### Hopper-v4 Milestone Trajectory

| Generation | Timesteps | Population Mean | Parent Eval Return | Best Return |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 9,050 | 146.4 | 112.8 | **112.8** |
| Gen 0038 | 1,714,707 | 979.5 | 1027.0 | **1028.5** |
| Gen 0075 | 3,968,548 | 990.5 | 1028.2 | **1029.0** |
| Gen 0112 | 6,042,422 | 1229.8 | 1727.1 | **1727.1** |
| Gen 0150 | 7,900,120 | 1783.2 | 1862.5 | **2045.1** |

### Swimmer-v4 Milestone Trajectory

| Generation | Timesteps | Population Mean | Parent Eval Return | Best Return |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 64,000 | -3.7 | -24.7 | **-24.7** |
| Gen 0038 | 2,432,000 | 346.1 | 349.2 | **354.6** |
| Gen 0075 | 4,800,000 | 357.6 | 354.2 | **359.2** |
| Gen 0112 | 7,168,000 | 357.9 | 355.0 | **359.6** |
| Gen 0150 | 9,600,000 | 358.2 | 359.4 | **359.6** |

### Walker2d-v4 Milestone Trajectory

| Generation | Timesteps | Population Mean | Parent Eval Return | Best Return |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 3,760 | 40.7 | 68.7 | **68.7** |
| Gen 0038 | 751,731 | 760.7 | 996.8 | **996.9** |
| Gen 0075 | 2,719,349 | 858.9 | 1056.4 | **1056.4** |
| Gen 0112 | 4,695,949 | 1013.2 | 1323.6 | **1323.6** |
| Gen 0150 | 6,750,131 | 1550.1 | 1919.5 | **1919.5** |
