"""Create and losslessly check Hub Parquet views from frozen release archives.

Requires numpy, pyarrow==19.0.1. The source directory contains publication.json
and its canonical <hf_config>.tar.gz archives. This script does not upload.
"""
import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xrdbench.hub import unpack_archive
from xrdbench.io import checked_load, file_hash, manifest, read_json, write_json


def arrow_array(values):
    if values.ndim == 2:
        return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), from_pandas=False), values.shape[1])
    return pa.array(values, from_pandas=False)


def paired_shards(root, meta, native_split):
    observations = meta["splits"][native_split]["observations"]
    target_root = root / ("private" if native_split == "test" else "public")
    targets = (read_json(target_root / "manifest.json")["test_targets"] if native_split == "test"
               else meta["splits"][native_split]["targets"])
    if len(observations) != len(targets):
        raise ValueError("Observation/target shard count mismatch")
    for obs_record, target_record in zip(observations, targets):
        obs = checked_load(root / "public", obs_record)
        labels = checked_load(target_root, target_record)
        for key in ("ids", "group_ids", "condition"):
            np.testing.assert_array_equal(obs[key], labels[key])
        yield {**obs, "theta": labels["theta"]}


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "archives").mkdir()
    catalog = read_json(source / "publication.json")
    total_rows = 0
    for entry in catalog["releases"]:
        config = entry["hf_config"]
        archive = source / Path(entry["archive"]["path"]).name
        destination = output / entry["archive"]["path"]
        shutil.copyfile(archive, destination)
        with tempfile.TemporaryDirectory(prefix="pxrd-parquet-") as temp:
            root = Path(temp) / "release"
            unpack_archive(destination, root, entry)
            meta = manifest(root / "public")
            metadata_dir = output / "metadata" / config
            metadata_dir.mkdir(parents=True)
            shutil.copyfile(root / "public/manifest.json", metadata_dir / "manifest.json")
            shutil.copyfile(root / "public" / meta["prototype"]["path"], metadata_dir / "prototype.cif")
            entry["metadata_path"] = f"metadata/{config}/manifest.json"
            entry["parquet"] = {}
            group_sets = {}
            for split, native in (("train", "train"), ("validation", "val"), ("test", "test")):
                path = output / "parquet" / config / f"{split}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                groups, ids, rows = set(), set(), 0
                writer = None
                try:
                    for arrays in paired_shards(root, meta, native):
                        table = pa.table({key: arrow_array(value) for key, value in arrays.items()})
                        if writer is None:
                            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                        writer.write_table(table, row_group_size=len(arrays["ids"]))
                        if ids.intersection(arrays["ids"].tolist()):
                            raise ValueError("Duplicate observation IDs")
                        ids.update(arrays["ids"].tolist())
                        groups.update(arrays["group_ids"].tolist())
                        rows += len(arrays["ids"])
                finally:
                    if writer is not None:
                        writer.close()
                expected = entry["sizes"][native]
                assert rows == expected["observation_count"]
                assert len(groups) == expected["structure_count"]
                # Read every output row back from disk and compare every field.
                parquet = pq.ParquetFile(path)
                parts = list(meta["splits"][native]["observations"])
                assert parquet.num_row_groups == len(parts)
                for i, arrays in enumerate(paired_shards(root, meta, native)):
                    actual = parquet.read_row_group(i)
                    assert set(actual.column_names) == set(arrays)
                    for key, expected_array in arrays.items():
                        column = actual[key].combine_chunks()
                        value = (column.values.to_numpy(zero_copy_only=False).reshape(expected_array.shape)
                                 if expected_array.ndim == 2 else column.to_numpy(zero_copy_only=False))
                        np.testing.assert_array_equal(value, expected_array)
                entry["parquet"][split] = {"path": path.relative_to(output).as_posix(),
                    "sha256": file_hash(path), "bytes": path.stat().st_size,
                    "rows": rows, "structures": len(groups), "all_columns_roundtrip_exact": True}
                group_sets[split] = groups
                total_rows += rows
            assert not group_sets["train"] & group_sets["validation"]
            assert not group_sets["train"] & group_sets["test"]
            assert not group_sets["validation"] & group_sets["test"]
        print(f"Verified {config}: all three splits, all labels, exact Parquet round trip", flush=True)
    catalog["stored_observation_rows"] = total_rows
    catalog["parquet_conversion"] = {"pyarrow": pa.__version__, "all_rows_all_columns_verified": True,
                                     "split_structure_disjointness_verified": True}
    write_json(output / "publication.json", catalog)
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
