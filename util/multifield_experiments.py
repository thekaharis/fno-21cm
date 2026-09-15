"""Plan staged mapping sweeps, run one job, and compare completed results.

Planning and summarizing use only the Python standard library. Execution uses
the Python interpreter selected with --python (the current interpreter by
default) and passes arguments directly, without shell interpolation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from dataset.fields import FOUR_FIELDS, FieldRegistry, enumerate_mappings


PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def sampling_argv(settings):
    flags = {"size": "window-size", "halo": "window-halo", "windows_per_cone": "windows-per-cone",
             "context_factor": "context-factor", "context_xy": "context-xy",
             "context_features": "context-features"}
    if set(settings) - (set(flags) | {"mode"}):
        raise ValueError("unknown sampling configuration option")
    mode = settings.get("mode", "full")
    if mode not in {"full", "contiguous", "coarse_context"}:
        raise ValueError("invalid sampling mode")
    result = ["--sampling", mode]
    for name, flag in flags.items():
        if name in settings:
            result.extend(("--"+flag, str(settings[name])))
    return result


def plan(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError(f"plan already exists: {out}")
    preparation_path = Path(args.preparation).resolve()
    prep = read_json(preparation_path)
    registry = FieldRegistry.from_dict(prep["registry"])
    mappings = enumerate_mappings(args.fields.split(","), args.stage, prep["conditioning"], registry)
    required = {name for mapping in mappings for name in mapping.fields}
    if required - set(prep["normalization"]):
        raise ValueError("prepare all sweep fields together before planning")
    seeds = args.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid run budget")
    settings = read_json(args.model_config) if args.model_config else {"kind": "localop", "ndim": 3}
    sampling = (read_json(args.sampling_config) if getattr(args, "sampling_config", None)
                else {"mode": "full"})
    sampling_flags = sampling_argv(sampling)
    if (prep["source"]["kind"] == "native") != (sampling.get("mode", "full") != "full"):
        raise ValueError("native preparation and window sampling must be used together")
    run_root = Path(args.run_root).resolve() if args.run_root else out.parent / "runs"
    entries = []
    preparation_digest = hashlib.sha256(preparation_path.read_bytes()).hexdigest()
    for mapping in mappings:
        for seed in seeds:
            parameters = {"mapping": mapping.to_dict(), "seed": seed,
                          "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
                          "model_settings": settings, "sampling": sampling,
                          "preparation_sha256": preparation_digest}
            digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:12]
            run_dir = run_root / mapping.slug / f"seed{seed}_{digest}"
            argv = [str(PROJECT / "fno_multifield.py"), "train", "--preparation", str(preparation_path),
                    "--inputs", ",".join(mapping.inputs), "--targets", ",".join(mapping.targets),
                    "--run-dir", str(run_dir), "--seed", str(seed), "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                    "--model-settings", json.dumps(settings, sort_keys=True), *sampling_flags]
            entries.append({"index": len(entries), **parameters, "run_dir": str(run_dir), "argv": argv})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "stage": args.stage,
                              "preparation": str(preparation_path),
                              "preparation_sha256": preparation_digest,
                              "mapping_count": len(mappings), "entries": entries}, indent=2) + "\n")
    print(f"Wrote {len(mappings)} mappings × {len(seeds)} seeds = {len(entries)} jobs to {out}")
    print(f"Job indices: 0–{len(entries)-1}; no training jobs have been submitted.")


def run(args):
    manifest = read_json(args.plan)
    current = hashlib.sha256(Path(manifest["preparation"]).read_bytes()).hexdigest()
    if current != manifest["preparation_sha256"]:
        raise ValueError("preparation artifact changed after this plan was written")
    if args.index < 0 or args.index >= len(manifest["entries"]):
        raise ValueError("job index is outside the plan")
    entry = manifest["entries"][args.index]
    metadata_path = Path(entry["run_dir"]) / "run_metadata.json"
    if args.skip_complete and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata["status"] == "complete" and (metadata_path.parent / "test_metrics.json").exists():
            print(f"Already complete: {entry['run_dir']}")
            return
    # The plan stores data, not arbitrary executable commands: reconstruct the
    # supported invocation from its declared fields before launching a run.
    mapping = entry["mapping"]
    command = [args.python, str(PROJECT / "fno_multifield.py"), "train",
               "--preparation", manifest["preparation"],
               "--inputs", ",".join(mapping["inputs"]), "--targets", ",".join(mapping["targets"]),
               "--run-dir", entry["run_dir"], "--seed", str(entry["seed"]),
               "--epochs", str(entry["epochs"]), "--batch-size", str(entry["batch_size"]),
               "--model-settings", json.dumps(entry["model_settings"]),
               "--workers", str(entry["workers"] if args.workers is None else args.workers),
               "--device", args.device, *sampling_argv(entry.get("sampling", {"mode": "full"}))]
    subprocess.run(command, cwd=PROJECT, check=True)


def comparison_key(metadata, target):
    training = metadata["training"]
    recipe = {key: training[key] for key in ("epochs", "seed", "batch_size", "learning_rate",
             "weight_decay", "grad_clip", "deterministic", "backbone_training")}
    # Checkpoint selection should be consistent, including whether it favors
    # a nominated primary target or the combined objective.
    recipe["monitor"] = training["monitor"]
    return json.dumps({"inputs": metadata["mapping"]["inputs"], "target": target,
                       "conditioning": metadata["mapping"]["conditioning"],
                       "model_config": metadata["model_config"], "training": recipe,
                       "sampling": metadata.get("sampling", {"mode": "full"}),
                       "preparation": metadata["preparation"]}, sort_keys=True)


def summarize(args):
    manifest = read_json(args.plan)
    records, incomplete = [], []
    for entry in manifest["entries"]:
        root = Path(entry["run_dir"])
        if not (root / "test_metrics.json").exists() or not (root / "run_metadata.json").exists():
            incomplete.append(entry["index"])
            continue
        metadata = read_json(root / "run_metadata.json")
        if metadata["status"] != "complete":
            incomplete.append(entry["index"])
            continue
        for target, metrics in read_json(root / "test_metrics.json")["fields"].items():
            records.append((metadata, target, metrics, str(root)))
    baselines = {comparison_key(meta, target): metrics["normalized_mse"]
                 for meta, target, metrics, _ in records if len(meta["mapping"]["targets"]) == 1}
    rows = []
    for meta, target, metrics, root in records:
        baseline = baselines.get(comparison_key(meta, target))
        row = {"inputs": ",".join(meta["mapping"]["inputs"]),
               "targets": ",".join(meta["mapping"]["targets"]), "field": target,
               "sampling": json.dumps(meta.get("sampling", {"mode": "full"}), sort_keys=True),
               "seed": meta["training"]["seed"], "run_dir": root,
               **{key: value for key, value in metrics.items() if not isinstance(value, dict)}}
        row["auxiliary_mse_gain"] = (1 - metrics["normalized_mse"]/baseline
            if len(meta["mapping"]["targets"]) > 1 and baseline is not None and baseline > 0 else None)
        rows.append(row)
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"summary directory already exists: {out}")
    out.mkdir(parents=True)
    if rows:
        with (out / "per_seed.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = {}
        for row in rows:
            groups.setdefault((row["inputs"], row["targets"], row["field"], row["sampling"]), []).append(row)
        with (out / "aggregate.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inputs", "targets", "field", "sampling", "n_seeds", "mean_normalized_mse",
                             "std_normalized_mse", "n_paired_seeds", "mean_auxiliary_mse_gain"])
            for key, group in groups.items():
                errors = [r["normalized_mse"] for r in group]
                gains = [r["auxiliary_mse_gain"] for r in group if r["auxiliary_mse_gain"] is not None]
                writer.writerow([*key, len(errors), statistics.mean(errors),
                                 statistics.stdev(errors) if len(errors) > 1 else "",
                                 len(gains), statistics.mean(gains) if gains else ""])
    (out / "status.json").write_text(json.dumps({"planned_jobs": len(manifest["entries"]),
                    "incomplete_job_indices": incomplete, "field_results": len(rows)}, indent=2) + "\n")
    print(f"Summarized {len(rows)} field results; {len(incomplete)} jobs incomplete. Output: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("plan")
    p.add_argument("--preparation", required=True)
    p.add_argument("--fields", default=",".join(FOUR_FIELDS))
    p.add_argument("--stage", choices=("pairwise", "single-target", "all"), default="pairwise")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--model-config")
    p.add_argument("--sampling-config", help="JSON window configuration (mode, size, halo, etc.)")
    p.add_argument("--run-root")
    p.add_argument("--out", required=True)
    p.set_defaults(function=plan)
    p = commands.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-complete", action="store_true")
    p.set_defaults(function=run)
    p = commands.add_parser("summarize")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
