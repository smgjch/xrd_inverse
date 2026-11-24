# v0.1 / v0.2 / 联合测试结果

每行对应一个提交及一个评测域。不同训练种子单独列出；95% CI 仅反映独立测试结构采样，不包含训练种子方差。
联合成绩按两个 ID 域等权，旧 OOD/stress 见 conditions.csv。训练版本未填写时标为 unspecified，不据此推断训练流程。

| 原型 / 赛道 | 方法 | 训练版本 / 种子 | 测试版本 | 平均相对误差 %（95% CI） | 0.5% 成功率 % | Energy ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| rutile / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1 | 0.0866 [0.0827, 0.0907] | 97.75 | 0.01030 |
| rutile / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.2 | 0.3769 [0.3551, 0.3993] | 76.46 | 0.04043 |
| rutile / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2318 [0.2205, 0.2434] | 87.11 | 0.02537 |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.1 | 2.5453 [2.4595, 2.6287] | 1.07 | 0.26255 |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.2 | 2.5449 [2.4582, 2.6268] | 1.30 | 0.26236 |
| rutile / blind | PriorSampler（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5451 [2.4590, 2.6269] | 1.19 | 0.26245 |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0987 [0.0953, 0.1024] | 98.31 | — |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.3827 [0.3600, 0.4051] | 75.65 | — |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2407 [0.2291, 0.2522] | 86.98 | — |
| rutile / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1 | 0.0937 [0.0877, 0.1001] | 96.45 | 0.01155 |
| rutile / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.2 | 0.3770 [0.3537, 0.4016] | 74.80 | 0.04017 |
| rutile / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2353 [0.2232, 0.2482] | 85.63 | 0.02586 |
| rutile / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.1 | 2.5453 [2.4595, 2.6287] | 1.07 | 0.26255 |
| rutile / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.2 | 2.5449 [2.4582, 2.6268] | 1.30 | 0.26236 |
| rutile / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5451 [2.4590, 2.6269] | 1.19 | 0.26245 |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0789 [0.0758, 0.0819] | 98.67 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.3289 [0.3057, 0.3524] | 78.12 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2039 [0.1920, 0.2161] | 88.40 | — |
| zno / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1 | 0.0834 [0.0786, 0.0892] | 97.53 | 0.00985 |
| zno / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.2 | 0.9301 [0.8760, 0.9841] | 53.26 | 0.08983 |
| zno / blind | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.5068 [0.4791, 0.5350] | 75.39 | 0.04984 |
| zno / blind | PriorSampler（无谱对照） | unspecified / None | v0.1 | 2.5718 [2.4851, 2.6588] | 0.81 | 0.26655 |
| zno / blind | PriorSampler（无谱对照） | unspecified / None | v0.2 | 2.5728 [2.4862, 2.6581] | 0.81 | 0.26690 |
| zno / blind | PriorSampler（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5723 [2.4855, 2.6579] | 0.81 | 0.26673 |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0858 [0.0818, 0.0906] | 98.76 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.9160 [0.8636, 0.9681] | 54.72 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.5009 [0.4735, 0.5277] | 76.74 | — |
| zno / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1 | 0.0850 [0.0778, 0.0939] | 97.14 | 0.01039 |
| zno / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.2 | 0.9152 [0.8609, 0.9687] | 54.26 | 0.08823 |
| zno / calibrated | Conditional-NSF-NPE | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.5001 [0.4709, 0.5292] | 75.70 | 0.04931 |
| zno / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.1 | 2.5718 [2.4851, 2.6588] | 0.81 | 0.26655 |
| zno / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.2 | 2.5728 [2.4862, 2.6581] | 0.81 | 0.26690 |
| zno / calibrated | PriorSampler（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5723 [2.4855, 2.6579] | 0.81 | 0.26673 |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0646 [0.0612, 0.0687] | 99.41 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.8950 [0.8422, 0.9461] | 56.74 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.4798 [0.4523, 0.5060] | 78.08 | — |
