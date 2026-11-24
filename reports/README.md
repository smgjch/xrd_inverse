# 结果与验证记录

benchmark v0.2 创建阶段仅交付数据与评测工具。后续增加了 [ConvNeXt1D-Lattice 点回归](convnext_pilot/README.md)及 [Conditional NSF / NPE 后验估计](nsf_pilot/README.md)的单 seed pilot 成绩：两个原型 × 两条赛道，各自使用联合版本训练，并分别报告 v0.1、v0.2 和联合测试域。NSF 报告包含与冻结 ConvNeXt 预测及无谱先验的配对比较；这些结果尚不能代表多种子、main 规模或真实实验数据上的排名。

`data_release_index.json` 登记共同数据版本；`tool_validation.json` 记录 benchmark 创建时的软件、兼容性、数据一致性检查。它们不是模型性能表，也不回写为后续训练的验证记录。

合作者完成模型推理并使用公开测试标签完成 `xrdbench score` 后，运行：

```bash
xrdbench summarize --scores /path/to/method_a/score /path/to/method_b/score \
  --output reports/collaborator_results
```

工具会生成 `matrix.csv`、`conditions.csv`、`results.json`、`report.md`。每份联合测试提交分成 v0.1、v0.2、等权联合三行；不同训练种子分开列出。没有收到的结果不会被填成零或用 prior 对照代替。

pilot 数据生成时冻结了生成器源码快照及依赖版本。数据登记中的 `generator_snapshot` 提供原快照的哈希；之后的文档和 evaluator 更新不回写生成历史。发布前的描述性元数据修订如有发生，以 `metadata_revision` 记录父数据 ID，观测文件保持不变。始终使用当前发布的 dataset_id 交付预测。

数据已发布到 [PXRD-LatticeBench](https://huggingface.co/datasets/smgjch/PXRD-LatticeBench)，包含完整 pilot 的 12 个配置及全部 train/validation/test 标签。`huggingface_release.json` 记录 Hub commit、文件验证及工具测试；与模型性能无关。数据下载方法见 [使用指南](../docs/数据下载与使用.md)。
