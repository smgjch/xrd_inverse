# v0.1 / v0.2 / 联合测试结果

每行对应一个提交及一个评测域。不同训练种子单独列出；95% CI 仅反映独立测试结构采样，不包含训练种子方差。
联合成绩按两个 ID 域等权，旧 OOD/stress 见 conditions.csv。训练版本未填写时标为 unspecified，不据此推断训练流程。

| 原型 / 赛道 | 方法 | 训练版本 / 种子 | 测试版本 | 平均相对误差 %（95% CI） | 0.5% 成功率 % | Energy ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0987 [0.0953, 0.1024] | 98.31 | — |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.3827 [0.3600, 0.4051] | 75.65 | — |
| rutile / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2407 [0.2291, 0.2522] | 86.98 | — |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / blind | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0789 [0.0758, 0.0819] | 98.67 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.3289 [0.3057, 0.3524] | 78.12 | — |
| rutile / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.2039 [0.1920, 0.2161] | 88.40 | — |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| rutile / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5309 [2.4440, 2.6139] | 1.56 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0858 [0.0818, 0.0906] | 98.76 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.9160 [0.8636, 0.9681] | 54.72 | — |
| zno / blind | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.5009 [0.4735, 0.5277] | 76.74 | — |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / blind | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1 | 0.0646 [0.0612, 0.0687] | 99.41 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.2 | 0.8950 [0.8422, 0.9461] | 56.74 | — |
| zno / calibrated | ConvNeXt1D-Lattice | v0.1+v0.2 / 0 | v0.1+v0.2 | 0.4798 [0.4523, 0.5060] | 78.08 | — |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
| zno / calibrated | PriorMean（无谱对照） | unspecified / None | v0.1+v0.2 | 2.5646 [2.4757, 2.6497] | 0.98 | — |
