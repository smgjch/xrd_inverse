"""Summarize collaborator predictions into matched v0.1/v0.2/union tables.

Consumes coordinator-scored artifacts only; never trains or generates predictions.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .io import load_npz, read_json, write_json
from .metrics import cluster_ci


def summarize_scores(scores, output):
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite report: {root}")
    rows, condition_rows, provenance = [], [], []
    identities, datasets = set(), {}
    for path in scores:
        path = Path(path)
        report = read_json(path / "metrics.json")
        raw = load_npz(path / "per_sample_metrics.npz")
        if report.get("suite") != "v0.1+v0.2" or set(report.get("by_domain", {})) != {"v0.1", "v0.2"}:
            raise ValueError("Summarize requires a scored union release for every checkpoint; export once on v0.1+v0.2")
        if report["validation"]["split"] != "test":
            raise ValueError("Performance reports require held-out test scores, not validation scores")
        scope = (report["task"], report["track"], report["validation"]["view"])
        if scope in datasets and datasets[scope] != report["dataset_id"]:
            raise ValueError("Methods in the same task/track/view must use the same release fingerprint")
        datasets[scope] = report["dataset_id"]
        metadata = report["method_metadata"]
        seed = metadata.get("training_seed", metadata.get("seed"))
        identity = {"task": report["task"], "track": report["track"],
                    "view": report["validation"]["view"], "method": metadata["method_name"],
                    "training_suite": metadata.get("training_suite", "unspecified"),
                    "training_seed": seed, "inference_seed": metadata.get("inference_seed", 0),
                    "purpose": metadata.get("purpose", "submitted_method")}
        key = tuple(identity.values())
        if key in identities:
            raise ValueError("Duplicate method/training/inference seed entry; do not count a checkpoint twice")
        identities.add(key)
        id_names = report["id_conditions"]
        old_names = [c for c in id_names if not c.startswith("hard_")]
        hard_names = [c for c in id_names if c.startswith("hard_")]
        if len(old_names) != 6 or len(hard_names) != 6:
            raise ValueError("Unexpected protocol domains: equal union weights require six ID conditions each")
        # A complete Cartesian set makes domain and structure bootstrap weights explicit.
        for name in id_names:
            chosen = raw["condition"] == name
            groups = raw["group_ids"][chosen]
            if len(groups) != len(set(groups.tolist())):
                raise ValueError("Repeated structure-condition row")
            if set(groups.tolist()) != set(raw["group_ids"].tolist()):
                raise ValueError("Unequal structure coverage across conditions")
        evaluated = {**report["by_domain"], "v0.1+v0.2": report["primary_id_macro"]}
        for domain, values in evaluated.items():
            names = old_names if domain == "v0.1" else hard_names if domain == "v0.2" else id_names
            selected = np.isin(raw["condition"], names)
            relative = raw["absolute_relative_error"][selected].mean(-1) * 100
            interval = cluster_ci(relative, raw["group_ids"][selected])
            rows.append({**identity, "evaluation_suite": domain,
                "n_structures": values["n_structures"], "n_observations": values["n_observations"],
                "mean_relative_error_percent": float(relative.mean()),
                "mre_ci95_low": interval["ci95"][0], "mre_ci95_high": interval["ci95"][1],
                "mae_angstrom_a": values["mae_angstrom"][0], "mae_angstrom_c": values["mae_angstrom"][1],
                "success_0.1pct_percent": 100 * values["success_rel_0.001"],
                "success_0.5pct_percent": 100 * values["success_rel_0.005"],
                "energy_score": values.get("energy_score", ""),
                "coverage90_mean": float(np.mean(values["marginal_coverage_0.9"])) if "marginal_coverage_0.9" in values else "",
                "width90_prior_fraction_mean": float(np.mean(values["interval_width_normalized_0.9"])) if "interval_width_normalized_0.9" in values else "",
                "failure_rate": values["failure_rate"], "dataset_id": report["dataset_id"],
            })
        for condition, values in report["by_condition"].items():
            condition_rows.append({**identity, "condition": condition,
                "in_primary": condition in id_names,
                "mean_relative_error_percent": float(np.mean(values["absolute_relative_error"]) * 100),
                "energy_score": values.get("energy_score", ""),
                "success_0.5pct_percent": values["success_rel_0.005"] * 100})
        provenance.append({**identity, "evaluation_dataset_id": report["dataset_id"],
            "training_dataset_id": metadata.get("training_dataset_id"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "submission_sha256": report["submission_sha256"], "preset": report["preset"],
            "inference_timing": {k: v for k, v in report["inference_timing"].items()
                                 if k not in ("host_platform", "batch_seconds", "batch_rows")}})
    if not rows:
        raise ValueError("No scored submissions supplied; no performance can be reported")
    root.mkdir(parents=True)
    for filename, table in [("matrix.csv", rows), ("conditions.csv", condition_rows)]:
        with open(root / filename, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    write_json(root / "results.json", {
        "combined_rule": "equal weights for v0.1/v0.2 ID domains; OOD/stress excluded",
        "ci": "95% bootstrap over independent structures, conditional on the fitted model; seeds remain separate rows",
        "provenance": provenance, "rows": rows,
    })
    lines = ["# v0.1 / v0.2 / 联合测试结果", "",
             "每行对应一个提交及一个评测域。不同训练种子单独列出；95% CI 仅反映独立测试结构采样，不包含训练种子方差。",
             "联合成绩按两个 ID 域等权，旧 OOD/stress 见 conditions.csv。训练版本未填写时标为 unspecified，不据此推断训练流程。",
             "",
             "| 原型 / 赛道 | 方法 | 训练版本 / 种子 | 测试版本 | 平均相对误差 %（95% CI） | 0.5% 成功率 % | Energy ↓ |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        energy = "—" if r["energy_score"] == "" else f"{r['energy_score']:.5f}"
        method = str(r["method"]).replace("|", "\\|").replace("\n", " ")
        if r["purpose"] == "negative_control":
            method += "（无谱对照）"
        lines.append(f"| {r['task']} / {r['track']} | {method} | {r['training_suite']} / {r['training_seed']} | {r['evaluation_suite']} | "
                     f"{r['mean_relative_error_percent']:.4f} [{r['mre_ci95_low']:.4f}, {r['mre_ci95_high']:.4f}] | "
                     f"{r['success_0.5pct_percent']:.2f} | {energy} |")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(root), "submissions": len(provenance), "matrix_rows": len(rows),
            "training_performed": False}
