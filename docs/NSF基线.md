# Conditional NSF / NPE 基线

本方法在 PXRD-LatticeBench 的固定训练对上，通过最大化条件密度训练 `p(theta | observed PXRD)`。它属于单轮、可摊销 Neural Posterior Estimation，不为单张测试谱额外模拟、顺序更新或重新训练。

首轮结果、训练曲线与校准分析见 [pilot 报告](../reports/nsf_pilot/README.md)。

论文：[Neural Spline Flows, Durkan et al., NeurIPS 2019](https://arxiv.org/abs/1906.04032)。Spline 公式改编自 [nflows](https://github.com/bayesiains/nflows/tree/5d22344436a4a5f4c3b93a3b99322a6bfe374c08)，保留 [MIT 许可证](../THIRD_PARTY_LICENSES/nflows.txt)。使用直接 PyTorch 实现，避免引入整套 sbi 运行依赖；这是该方法在当前任务上的迁移，不是复现某篇 XRD 论文的原始成绩。

## 模型与概率定义

- 使用与 [ConvNeXt 基线](ConvNeXt基线.md) 相同的 stem、三个 stage、保留角度位置的池化、元数据处理和 512→128 特征投影。全部从零训练，没有加载 ConvNeXt checkpoint。
- 输入为谱、mask、固定角度坐标及七维公共元数据。blind 的 nuisance 真值始终不可读；`theta`、condition 和 group ID 不作为推理输入。
- 将 `(a,c)` 用冻结的先验上下界变换到 `[0,1]^2`。
- 六层交替 scalar coupling，每层使用八个 rational-quadratic spline bins，条件 MLP 宽度 128；一个坐标的 spline 由另一个坐标和谱特征决定。
- Flow 将单位参数空间双射到 `Uniform([0,1]^2)` base；正向 Jacobian 给出归一化条件密度，逆变换给出后验样本。零初始化对应均匀先验。
- 这是有界 NSF。数学上的支持集由变换本身保证，不通过事后裁剪或未归一化截断得到。FP32 逆变换在边界可能出现浮点舍入误差；evaluator 如实统计越界样本，数值审计记录最大偏移，没有事后裁剪。报告的 NLL 在先验归一化坐标下；换成 Å 坐标的 log density 时需减去 `log(prod(upper-lower))`，对应 NLL 加上此项。
- 采样种子由统一 runner 的 observation ID 和 inference seed 给出；输出 `[B,256,2]` Å 样本，保持样本前缀与 batch 切分一致。没有 Physics loss、候选重排或局部精修。

## 首轮 pilot 配方

分别在 rutile / ZnO × blind / calibrated 的联合 `v0.1+v0.2` 数据上训练，seed=0。每次使用 5000 个独立训练结构、10000 张谱；validation 为 512 个独立结构、6144 张谱。train/validation/test 使用冻结划分，不增加模拟或随机谱增强。

默认最多 100 epoch、batch=128、FP32、AdamW、学习率 3e-4、weight decay=1e-4、梯度范数上限 5；3 epoch warmup 后 cosine 衰减。每次拟合默认 3500 秒上限。第 1 epoch、随后每 5 epoch、最后一个 epoch 进行完整 validation。

**只按 validation energy score 选择 checkpoint。** 每张 validation 谱使用 256 个固定 base uniforms，通过共同随机数减少不同 checkpoint 比较时的 Monte Carlo 波动；样本与验证标签独立。同步记录 normalized MAE 和 exact NLL。30 epoch 后，连续八次 validation 无改善时允许早停。评分器重新导出 validation 时使用统一 inference seed，其 Monte Carlo 数值可与训练期选择日志略有不同。

NPE 与点回归使用相同数据预算、编码器及训练时限；输出头、训练目标、学习率和 checkpoint 选择指标按方法分别登记。记录真实训练与采样成本，不能以同样 epoch 数声称计算量完全相等。

## 运行与统一评测

使用匹配的 PyTorch/torch_npu/CANN 环境。CPU 可安装 `pip install -e '.[baseline]'`；不要在平台 Ascend 镜像内用这个命令覆盖配套 torch。实际加速器操作通过获授权的任务平台提交。

```bash
python -m examples.smoke_nsf --device npu:0 --output outputs/nsf-smoke

python -m xrdbench.train_nsf \
  --public data/rutile/blind/v0.1+v0.2/public \
  --output outputs/nsf/rutile/blind/seed0 \
  --device npu:0 --epochs 100 --batch-size 128 --seed 0 --export-test
```

训练只读取 train/val；`--export-test` 在 checkpoint 选择完成后导出统一格式的 validation/test 后验预测，不读取测试标签。输出包含 `best.pt`、`method.json`、`history.json`、`training_summary.json`、环境版本以及两个 prediction 目录。

```bash
python -m examples.evaluate_posterior_matrix \
  --data-root data --run-root outputs/nsf \
  --point-run-root outputs/convnext \
  --output outputs/nsf-evaluation
```

四个模型按 `<run-root>/<task>/<track>/seed0` 组织。`--point-run-root` 为可选项，用于加载已经冻结的 ConvNeXt 点预测；评测命令不训练或修改它们。

评测输出 NSF、PriorSampler、PriorMean，以及可选 ConvNeXt 的匹配结果。每个 checkpoint 的一次联合 test 推理产生 v0.1、v0.2、联合三行主成绩；17 个条件均保留，OOD/stress 单列。报告 energy score、覆盖率与宽度、SBC ranks、样本边缘中位数的点误差、成功率、成本，以及结构配对 bootstrap。

PriorSampler 的 energy/coverage 是无谱参照；ConvNeXt 只参与点指标比较。Oracle@K 不能代替实际点预测。纯无信号条件没有专用解析 fallback，需要在 OOD/stress 中如实检查模型是否过度自信。

## 验证与限制

自动测试覆盖单 spline 与组合 flow 的正反变换、autograd Jacobian、密度归一化、先验初始化、支持集、条件密度学习、energy 计算、种子一致性、无测试标签训练和统一 posterior 评分。NPU 调试另外检查完整网络反向传播和每谱 256 个样本的逆变换。

首轮是单 seed synthetic pilot。数值物理后验仍需独立收敛认证，不能由 NSF 的结果推断其为真后验；点误差更小也不自动证明校准更好。不同架构、超参或 posterior sample 数的扩展实验应单独登记。
