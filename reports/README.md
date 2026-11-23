# 结果与验证记录

当前交付的是 benchmark v0.2、pilot 数据与评测工具。**尚无合作者提交的已训练 baseline 成绩，本轮没有训练模型或提交加速卡任务。**

`data_release_index.json` 登记共同数据版本；`tool_validation.json` 记录软件、兼容性、数据一致性检查。它们不是模型性能表。

合作者完成模型推理、协调者完成 `xrdbench score` 后，运行：

```bash
xrdbench summarize --scores /path/to/method_a/score /path/to/method_b/score \
  --output reports/collaborator_results
```

工具会生成 `matrix.csv`、`conditions.csv`、`results.json`、`report.md`。每份联合测试提交分成 v0.1、v0.2、等权联合三行；不同训练种子分开列出。没有收到的结果不会被填成零或用 prior 对照代替。

pilot 数据生成时冻结了生成器源码快照及依赖版本。数据登记中的 `generator_snapshot` 提供原快照的哈希；之后的文档和 evaluator 更新不回写生成历史。发布前的描述性元数据修订如有发生，以 `metadata_revision` 记录父数据 ID，观测文件保持不变。始终使用当前发布的 dataset_id 交付预测。
