from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual eval for all runs and append history per run.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=ROOT / ".venv" / "bin" / "python",
        help="Python interpreter used to launch eval scripts.",
    )
    parser.add_argument(
        "--prefer-best-model",
        action="store_true",
        help="Use eval/best_model.zip before catchme_ppo.zip when both are present.",
    )
    return parser.parse_args()


def pick_model(run_dir: Path, prefer_best: bool) -> Path | None:
    final_model = run_dir / "catchme_ppo.zip"
    best_model = run_dir / "eval" / "best_model.zip"

    candidates = [best_model, final_model] if prefer_best else [final_model, best_model]
    for model_path in candidates:
        if model_path.exists():
            return model_path
    return None


def infer_eval_mode(run_dir: Path) -> str:
    if "continuous" in run_dir.name.lower():
        return "continuous"
    return "discrete"


def run_eval(
    *,
    python_bin: Path,
    model_path: Path,
    run_dir: Path,
    episodes: int,
    mode: str,
) -> tuple[bool, str]:
    module_name = "eval_continuous" if mode == "continuous" else "eval"
    cmd = [
        str(python_bin),
        "-m",
        module_name,
        "--model-path",
        str(model_path),
        "--run-dir",
        str(run_dir),
        "--episodes",
        str(episodes),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, output


def main() -> None:
    args = parse_args()
    if not args.runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {args.runs_dir}")
    if not args.python_bin.exists():
        raise FileNotFoundError(f"Python binary not found: {args.python_bin}")

    run_dirs = sorted([p for p in args.runs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        print("No run directories found.")
        return

    started = time.time()
    total = len(run_dirs)
    attempted = 0
    succeeded = 0
    skipped = 0
    failed = 0
    failures: list[str] = []

    print(f"Found {total} run directories. Episodes per run: {args.episodes}")
    for idx, run_dir in enumerate(run_dirs, start=1):
        model_path = pick_model(run_dir, prefer_best=args.prefer_best_model)
        if model_path is None:
            skipped += 1
            print(f"[{idx:02d}/{total:02d}] SKIP {run_dir.name}: no model found")
            continue

        attempted += 1
        mode = infer_eval_mode(run_dir)
        print(f"[{idx:02d}/{total:02d}] RUN  {run_dir.name}: mode={mode}, model={model_path.relative_to(ROOT)}")

        ok, output = run_eval(
            python_bin=args.python_bin,
            model_path=model_path,
            run_dir=run_dir,
            episodes=args.episodes,
            mode=mode,
        )
        if not ok:
            # Fallback to alternate evaluator in case mode inference is wrong.
            alt_mode = "continuous" if mode == "discrete" else "discrete"
            ok, output = run_eval(
                python_bin=args.python_bin,
                model_path=model_path,
                run_dir=run_dir,
                episodes=args.episodes,
                mode=alt_mode,
            )

        if ok:
            succeeded += 1
            print(f"[{idx:02d}/{total:02d}] OK   {run_dir.name}")
        else:
            failed += 1
            failures.append(run_dir.name)
            print(f"[{idx:02d}/{total:02d}] FAIL {run_dir.name}")
            print(output.strip()[-1200:])

    elapsed = time.time() - started
    print("\nBatch Eval Summary")
    print(f"attempted: {attempted}")
    print(f"succeeded: {succeeded}")
    print(f"failed:    {failed}")
    print(f"skipped:   {skipped}")
    print(f"elapsed_s: {elapsed:.2f}")
    if failures:
        print(f"failed_runs: {', '.join(failures)}")

    # Non-zero exit on failures helps automation.
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
