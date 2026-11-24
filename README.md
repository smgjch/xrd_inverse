# PXRD-LatticeBench

统一的模拟粉末 XRD 晶格参数反演任务、baseline 接入与独立评测工具。已知晶体原型，估计连续晶格参数；全部数据为模拟数据。

v0.2 加入固定有限角度窗口、低计数与未知仪器参数，保留 v0.1。支持用**一份联合测试预测**分别汇报 v0.1、v0.2、等权联合成绩。

本仓库不替合作者训练模型，也不把接口验证当作 baseline 性能。

**完整 pilot 数据（含 train / validation / test 和全部标签）已发布至 [Hugging Face](https://huggingface.co/datasets/smgjch/PXRD-LatticeBench)。** 数据版本为 `v0.2-pilot`，工具版本为 `0.2.1`。GitHub 保存源码与协议，数据文件从 Hugging Face 下载。

- [v0.2 实验协议与接入流程](docs/v0.2实验协议.md)
- [v0.1 完整实验协议、指标与方法迁移指南](docs/实验协议与接入指南.md)
- [方法元数据模板](examples/method_metadata.json)
- [离线预测导出](examples/export_npz.py)
- [数据下载、字段说明与 Hugging Face 用法](docs/数据下载与使用.md)

## 安装

Python 3.11，CPU evaluator 无 PyTorch 依赖。原方法可以保留自己的训练环境。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,hub]'
pytest -q
```

重建冻结的 Linux 评分依赖可用 `requirements-evaluator.lock.txt`。数据以协调者发布的文件和 dataset_id 为准，不要求合作者重复模拟。

## 下载统一数据

```bash
# 推荐：下载一个联合版本，包含训练、验证、测试谱及全部标签。
xrdbench download --task rutile --track blind --suite v0.1+v0.2 \
  --output data/rutile/blind/v0.1+v0.2

# 全部 12 个配置：2 原型 × 2 赛道 × 3 版本。
xrdbench download --all --output data
```

默认固定到本次发布的 Hugging Face commit，自动核对压缩包 SHA-256、manifest 指纹和每个原始文件的 SHA-256。可用 `--cache-dir` 指定云盘缓存。公开数据无需 token。下载失败可重新运行；已有且完整一致的目标目录会复用，不覆盖其他数据。

| 每个原型、赛道下的版本 | train 谱数 | validation 谱数 | test 谱数 |
| --- | ---: | ---: | ---: |
| v0.1 | 10,000 | 3,072 | 5,632 |
| v0.2 | 10,000 | 3,072 | 3,072 |
| v0.1+v0.2 | 10,000 | 6,144 | 8,704 |

每个配置均为 5,000 / 512 / 512 个独立结构。各版本与赛道共享部分结构或观测，不能将 12 个配置当成独立样本直接拼接。原始目录 `public/val` 对应 Hugging Face 的 `validation`；`private/test` 是兼容旧工具的目录名，其测试标签在本次发布中完全公开。训练与调参仍只使用 train/validation，test 仅供最终评估。

## 十分钟内验证接入

下面运行无谱 prior 对照，只验证流程，不产生已训练模型成绩。命令拒绝覆盖已有输出目录。

```bash
XRD_WORK="$PWD/outputs/demo-v02"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

xrdbench generate --task rutile --track blind --suite v0.1+v0.2 \
  --preset smoke --output "$XRD_WORK/data"
xrdbench audit --public "$XRD_WORK/data/public" --private "$XRD_WORK/data/private"
xrdbench run --public "$XRD_WORK/data/public" --adapter xrdbench.api:PriorSampler \
  --kind posterior --output "$XRD_WORK/prior_prediction"
xrdbench score --public "$XRD_WORK/data/public" --private "$XRD_WORK/data/private" \
  --submission "$XRD_WORK/prior_prediction" --output "$XRD_WORK/prior_score"
xrdbench summarize --scores "$XRD_WORK/prior_score" --output "$XRD_WORK/report"
```

报告生成 `report.md`、`matrix.csv`、`conditions.csv`、`results.json`。每份联合测试提交分成三个评测域，联合主成绩对旧/新 ID 条件各给 50% 权重；旧 OOD/stress 单列。

## 合作者流程

1. 用 `xrdbench download` 获取统一数据，记录 dataset_id；使用 train 训练，validation 选择超参数和 checkpoint。
2. 实现 `Adapter(context, checkpoint)`、`predict(batch, n_samples, seeds)` 和 `synchronize()`，或在原环境导出 NPZ。
3. 在对应原型/赛道的联合 public/test 上导出一次预测，用 `xrdbench validate` 检查。
4. 使用下载包中公开的 private/test 标签执行 `score`，再用 `summarize --scores ...` 汇总多方法、多种子结果。

预测类型为 point、posterior（256 个原始样本）或 candidates（64 个候选和选定点）。模型不能读取测试标签、干净谱或隐藏 nuisance。精修/重排后的集合不得冒充原始后验。

PyTorch 接线示例见 [torch_point_adapter.py](examples/torch_point_adapter.py)，它不是 DeepLPnet/AlphaDiffract 的复现。对输入、先验和输出约定有疑问时以协议文档为准。

## 数据与发布

```bash
# 维护者生成新版本时使用；标准比较请直接下载冻结发布，不重新生成测试数据。
python -m examples.generate_matrix --task rutile --preset pilot --output "$XRD_DATA_ROOT"
python -m examples.generate_matrix --task zno --preset pilot --output "$XRD_DATA_ROOT"
```

目录为 `<data-root>/<task>/<track>/<suite>/{public,private}`。三套训练配方都保持每结构两张谱；联合版各取一张旧/新域观测。比较训练配方是可选实验，不是生成三个测试分数的必要条件。

`.gitignore` 排除 NPZ/Parquet/Arrow 数据、压缩包、checkpoint、预测、环境、缓存、日志、密钥和私有部署记录；保留源码、文档和可公开的小型汇总 CSV/JSON。数据集沿用 Hugging Face 仓库的 Apache-2.0 许可证。仓库不包含训练权重或真实实验 XRD。

## 边界

blind 的原始参数/后验评测已支持；公共物理候选评分目前仅支持 calibrated。粗网格参考不是认证的精确后验。新增窗口旨在增加有限信息与参数耦合，不能仅凭窗口设计宣称所有观测都是多峰。当前没有真实实验谱泛化或 baseline 排名结论。
