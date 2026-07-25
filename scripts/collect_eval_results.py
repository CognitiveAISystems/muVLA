#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


TASK_ORDER = [
    "ShellGameTouch-VLA-v0",
    "ShellGamePush-VLA-v0",
    "ShellGamePick-VLA-v0",
    "InterceptSlow-VLA-v0",
    "InterceptMedium-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
    "TakeItBack-VLA-v0",
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
]

# `_+` rather than `_`: the launchers join the checkpoint tag and --results_note with
# an underscore, so the documented `--results_note "_rh-true"` produces two of them.
RUN_DIR_RE = re.compile(r"(?P<exp_name>.+)--(?P<ckpt>\d+)_chkpt_+rh-(?P<rh>true|false)$")
SUCCESS_RATE_RE = re.compile(r"Success rate:\s*(?P<rate>\d+(?:\.\d+)?)\s*±")


@dataclass(frozen=True)
class RunColumn:
    exp_name: str
    ckpt_raw: int
    rh: bool
    run_dir: Path

    @property
    def ckpt_label(self) -> str:
        if self.ckpt_raw % 1000 == 0:
            return f"{self.ckpt_raw // 1000}k"
        return str(self.ckpt_raw)

    @property
    def rh_label(self) -> str:
        return "TRUE" if self.rh else "FALSE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect evaluation success rates into a CSV table.")
    parser.add_argument(
        "--eval-results-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory with per-run evaluation results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results_summary.csv"),
        help="Path to the output CSV file.",
    )
    return parser.parse_args()


def discover_runs(eval_results_dir: Path) -> list[RunColumn]:
    runs: list[RunColumn] = []
    for child in sorted(eval_results_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_RE.search(child.name)
        if not match:
            continue
        runs.append(
            RunColumn(
                exp_name=match.group("exp_name"),
                ckpt_raw=int(match.group("ckpt")),
                rh=match.group("rh") == "true",
                run_dir=child,
            )
        )
    runs.sort(key=lambda run: (extract_exp_id(run.exp_name), run.exp_name, -run.ckpt_raw, not run.rh))
    return runs


def extract_exp_id(exp_name: str) -> int:
    match = re.search(r"exp_id_(\d+)", exp_name)
    return int(match.group(1)) if match else 10**9


def find_latest_log(run_dir: Path, task_name: str) -> Path | None:
    task_dir = run_dir / task_name
    if not task_dir.exists():
        return None
    logs = sorted(task_dir.glob("**/logs/EVAL-*.txt"))
    return logs[-1] if logs else None


def parse_success_rate(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = SUCCESS_RATE_RE.findall(text)
    if not matches:
        return "TBA"
    return f"{float(matches[-1]):.2f}"


def build_rows(runs: list[RunColumn]) -> list[list[str]]:
    rows: list[list[str]] = []
    rows.append(["Exp name", *[run.exp_name for run in runs]])
    rows.append(["CKPT", *[run.ckpt_label for run in runs]])
    rows.append(["Receding Horizon", *[run.rh_label for run in runs]])

    for task_name in TASK_ORDER:
        row = [task_name]
        for run in runs:
            log_path = find_latest_log(run.run_dir, task_name)
            row.append(parse_success_rate(log_path) if log_path else "TBA")
        rows.append(row)
    return rows


def write_csv(rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.eval_results_dir)
    rows = build_rows(runs)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows) - 3} task rows for {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
