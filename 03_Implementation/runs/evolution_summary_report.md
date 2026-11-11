# OpenAI Evolution Strategies (ES) — 30-Seed Statistical Benchmark Report

## 1. Multi-Seed Statistical Performance Summary (`N=30` Independent Runs)

| Experiment / Environment | Algo | Action Space | Seeds ($N$) | Budget (Evals / Steps) | Mean ± Std Return | Median (IQR) | Best / Worst | Paper Target | Mean Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **halfcheetah** (`HalfCheetah-v4`) | `es` | Continuous (tanh) | 30 | 9,600 / 9,600,000 | **4890.0 ± 926.4** | 5204.5 (1234.9) | 6199.8 / 2721.4 | ~3000 | 118.2s |
| **hopper** (`Hopper-v4`) | `es` | 10 Bins | 30 | 9,600 / 8,202,881 | **1316.9 ± 437.1** | 1194.7 (43.1) | 2884.1 / 1028.1 | ~3000 | 195.2s |
| **swimmer** (`Swimmer-v4`) | `es` | 10 Bins | 30 | 9,600 / 9,600,000 | **244.2 ± 106.9** | 305.9 (192.2) | 363.9 / 52.9 | ~365 | 149.5s |
| **walker2d** (`Walker2d-v4`) | `es` | Continuous (tanh) | 30 | 9,600 / 7,304,445 | **1977.7 ± 667.4** | 1745.1 (1100.6) | 3391.3 / 1004.9 | ~2500 | 169.7s |

## 2. Per-Generation Convergence Milestones

### halfcheetah (`HalfCheetah-v4`, N=30 Seeds)

| Generation | Mean Timesteps | Eval Return (Mean ± Std) | Eval Return (Median / IQR) | Best Return (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 64,000 | -0.1 ± 2.2 | -0.3 (1.4) | **-0.1 ± 2.2** |
| Gen 0038 | 2,432,000 | 2785.2 ± 583.3 | 2781.5 (734.6) | **2827.4 ± 584.8** |
| Gen 0075 | 4,800,000 | 3650.6 ± 1070.8 | 3596.6 (1536.5) | **3830.9 ± 815.4** |
| Gen 0112 | 7,168,000 | 4354.3 ± 933.6 | 4604.1 (1259.1) | **4467.6 ± 908.5** |
| Gen 0150 | 9,600,000 | 4430.0 ± 1532.7 | 4914.8 (1213.3) | **4890.0 ± 926.4** |

### hopper (`Hopper-v4`, N=30 Seeds)

| Generation | Mean Timesteps | Eval Return (Mean ± Std) | Eval Return (Median / IQR) | Best Return (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 8,697 | 162.7 ± 52.1 | 144.8 (59.1) | **162.7 ± 52.1** |
| Gen 0038 | 1,475,938 | 1040.3 ± 80.1 | 1051.1 (35.0) | **1067.8 ± 42.7** |
| Gen 0075 | 3,693,209 | 1122.8 ± 91.4 | 1112.5 (85.2) | **1140.8 ± 91.4** |
| Gen 0112 | 5,914,828 | 1208.1 ± 295.4 | 1153.2 (67.6) | **1233.1 ± 296.1** |
| Gen 0150 | 8,202,881 | 1291.1 ± 424.3 | 1173.3 (41.9) | **1316.9 ± 437.1** |

### swimmer (`Swimmer-v4`, N=30 Seeds)

| Generation | Mean Timesteps | Eval Return (Mean ± Std) | Eval Return (Median / IQR) | Best Return (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 64,000 | 3.0 ± 19.6 | 10.5 (37.5) | **3.0 ± 19.6** |
| Gen 0038 | 2,432,000 | 131.8 ± 102.9 | 87.4 (128.3) | **136.1 ± 101.7** |
| Gen 0075 | 4,800,000 | 217.0 ± 104.0 | 245.0 (189.7) | **221.8 ± 104.8** |
| Gen 0112 | 7,168,000 | 233.7 ± 106.6 | 293.1 (193.0) | **238.4 ± 107.0** |
| Gen 0150 | 9,600,000 | 239.4 ± 106.0 | 297.0 (187.2) | **244.2 ± 106.9** |

### walker2d (`Walker2d-v4`, N=30 Seeds)

| Generation | Mean Timesteps | Eval Return (Mean ± Std) | Eval Return (Median / IQR) | Best Return (Mean ± Std) |
| :---: | :---: | :---: | :---: | :---: |
| Gen 0001 | 3,549 | 53.9 ± 38.4 | 48.9 (20.6) | **53.9 ± 38.4** |
| Gen 0038 | 961,099 | 954.4 ± 212.8 | 1017.4 (55.8) | **968.7 ± 211.3** |
| Gen 0075 | 2,972,195 | 1206.6 ± 247.0 | 1100.8 (162.5) | **1257.0 ± 265.6** |
| Gen 0112 | 5,102,018 | 1505.2 ± 519.9 | 1295.4 (797.4) | **1580.6 ± 528.5** |
| Gen 0150 | 7,304,445 | 1869.9 ± 691.0 | 1683.2 (1066.5) | **1977.7 ± 667.4** |
