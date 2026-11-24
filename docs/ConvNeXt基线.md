# ConvNeXt1D-Lattice 回归基线

这是在 PXRD-LatticeBench 上重新训练的 ConvNeXt1D 架构迁移。ConvNeXt 残差块参考 [OpenAlphaDiffract](https://github.com/AdvancedPhotonSource/OpenAlphaDiffract/tree/ec118b9d899f57e8971a7ad4d4aa23b3f38572e0)，保留其 BSD 声明并加入 channel LayerNorm；其余 stem、元数据处理和回归头按当前任务实现。相关论文：[ConvNeXt](https://arxiv.org/abs/2201.03545)、[AlphaDiffract](https://arxiv.org/abs/2603.23367)。本实现不复现原论文完整的晶系/空间群分类任务、数据或预训练成绩。

## 模型与输入

- 三通道：协议的 `log1p(counts)/log1p(exposure)`、观测 mask、固定绝对角度坐标。缺测计数保持零。
- 七维公共元数据：变换后的曝光、nuisance 是否已知，以及五个按固定尺度变换的 nuisance。blind 的未知值和对应变换特征均置零，保留未知标志；不拟合测试统计量。
- Conv stem：kernel=9、stride=4；三个 stage 的宽度 `[48,96,192]`、深度 `[2,2,4]`，stage 间 stride=2 下采样。
- 残差块：kernel=7 depthwise convolution、LayerNorm、4 倍展开 MLP、GELU、layer scale 和 stochastic depth。
- 输出保留 16 个角度位置的池化特征并 flatten，避免仅做全局平均而丢失峰位置；元数据通过 32 维 MLP 后拼接。
- 回归头：512→128→2，sigmoid 输出先验内的单位参数，再还原为 Å 单位 `(a,c)`。
- 不使用额外模拟、外部预训练或物理后处理。`condition`、`group_ids`、真值参数和隐藏 nuisance 均不作为推理输入。

默认模型有 **3,085,282** 个可训练参数。

## Pilot 配方

默认对每个原型/赛道分别训练：联合 `v0.1+v0.2` 的 5000 个独立结构、10000 条固定训练谱，validation 为 512 个结构、6144 条谱。保持官方划分，不增加随机谱增强。

第一轮为 seed=0，100 epoch 上限，batch=128，FP32；AdamW，初始目标学习率 5e-4、weight decay=0.01，3 epoch warmup 后 cosine decay；SmoothL1 loss，先验归一化空间 beta=0.02，梯度范数裁剪 1.0。head dropout=0.1，最大 drop path=0.05，layer scale 初值 1e-3。

按 validation 的平均 normalized MAE 保存最佳 checkpoint。30 epoch 后允许连续 20 个 epoch 无改善时早停；单次训练默认 3500 秒上限。训练曲线、时间与停止原因完整保存。时间上限或 validation patience 不构成充分收敛的证明；第一轮成绩是单 seed pilot，不作为论文最终排名。

## 运行

在原有匹配的 PyTorch / torch_npu / CANN 环境使用代码；CPU 环境可安装 `pip install -e '.[baseline]'`，不要在 Ascend 镜像中用它覆盖平台配套的 torch。实际加速器训练应通过已授权的任务平台提交。

```bash
python -m examples.smoke_convnext --device npu:0 --output outputs/convnext-smoke

python -m xrdbench.train_convnext \
  --public data/rutile/blind/v0.1+v0.2/public \
  --output outputs/convnext/rutile/blind/seed0 \
  --device npu:0 --epochs 100 --batch-size 128 --seed 0 --export-test
```

`--export-test` 在训练完成并固定 checkpoint 后，使用统一 adapter 导出 `prediction_val` 和 `prediction_test`。训练函数只加载 train/val，绝不加载 private/test 标签。输出包含 `best.pt`、`method.json`、`history.json`、`training_summary.json` 和环境版本。checkpoint 同时绑定原型、赛道、参数顺序、先验、谱轴和允许的评测 dataset_id。

```bash
xrdbench score --public data/rutile/blind/v0.1+v0.2/public \
  --private data/rutile/blind/v0.1+v0.2/private \
  --submission outputs/convnext/rutile/blind/seed0/prediction_test \
  --output outputs/convnext/rutile/blind/seed0/score_test
xrdbench summarize --scores outputs/convnext/rutile/blind/seed0/score_test \
  --output reports/convnext_pilot
```

一个联合 test 提交生成 v0.1、v0.2、联合三行主成绩。OOD/stress 保留在条件表中；对照 prior mean；另保存 validation 评分和学习曲线以判断过拟合/未收敛。盲赛道的点预测评测不依赖尚未实现的物理 nuisance 边缘化。

四个 seed=0 训练输出按 `<run-root>/<task>/<track>/seed0` 放置后，可以在 CPU 评测环境一次完成全部评分和 prior-mean 对照：

```bash
python -m examples.evaluate_point_matrix \
  --data-root data --run-root outputs/convnext \
  --output outputs/convnext-evaluation
```

其中 `data` 使用 `xrdbench download --all --output data` 的目录结构。此命令读取已冻结的预测，不训练模型；会生成 validation 评分、四组 test 成绩、同数据上的 prior 对照，以及三个 ID 域的结构配对 bootstrap 比较。汇总位于输出目录的 `summary/`。各输出目录必须尚不存在，避免覆盖已有实验。

首轮结果、训练历史和成本见 [pilot 报告](../reports/convnext_pilot/README.md)。报告中的 v0.1/v0.2/联合均为评测域；本轮只采用联合训练配方，没有运行三种训练配方的 3×3 扩展实验。
