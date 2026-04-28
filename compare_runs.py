import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_last(series: pd.Series, precision: int = 4):
    numeric = _to_float(series).dropna()
    if numeric.empty:
        return "N/A"
    return round(float(numeric.iloc[-1]), precision)


def _safe_last_int(series: pd.Series):
    numeric = _to_float(series).dropna()
    if numeric.empty:
        return "N/A"
    return f"{int(numeric.iloc[-1]):,}"


def _load_eval_metrics(eval_npz_path: Path) -> dict[str, object]:
    if not eval_npz_path.exists():
        return {
            "Best Eval Mean Reward": "N/A",
            "Latest Eval Mean Reward": "N/A",
            "Eval Count": 0,
        }

    try:
        data = np.load(eval_npz_path)
        results = data.get("results")
        if results is None or results.size == 0:
            return {
                "Best Eval Mean Reward": "N/A",
                "Latest Eval Mean Reward": "N/A",
                "Eval Count": 0,
            }

        eval_means = np.mean(results, axis=1)
        return {
            "Best Eval Mean Reward": round(float(np.max(eval_means)), 2),
            "Latest Eval Mean Reward": round(float(eval_means[-1]), 2),
            "Eval Count": int(len(eval_means)),
        }
    except Exception as e:
        print(f"Error reading {eval_npz_path}: {e}")
        return {
            "Best Eval Mean Reward": "N/A",
            "Latest Eval Mean Reward": "N/A",
            "Eval Count": 0,
        }


def collect_run_data(runs_root: Path):
    run_data = []

    if not runs_root.exists():
        print(f"Directory {runs_root} does not exist.")
        return []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        progress_path = run_dir / "progress.csv"
        eval_dir = run_dir / "eval"
        eval_npz_path = eval_dir / "evaluations.npz"
        best_model_path = eval_dir / "best_model.zip"
        checkpoint_dir = run_dir / "checkpoints"

        if not config_path.exists():
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        run_name = config.get("run_name")
        if run_name is None:
            run_name = config.get("run-name")
        if run_name in (None, ""):
            run_name = run_dir.name

        metrics = {
            "Run Name": run_name,
            "Run Dir": run_dir.name,
            "Learning Rate": config.get("learning_rate", "N/A"),
            "Batch Size": config.get("batch_size", "N/A"),
            "Steps": config.get("n_steps", "N/A"),
            "Ent Coef": config.get("ent_coef", "N/A"),
            "Gamma": config.get("gamma", "N/A"),
            "N Envs": config.get("n_envs", "N/A"),
            "Final Model": "yes" if (run_dir / "catchme_ppo.zip").exists() else "no",
            "Best Model": "yes" if best_model_path.exists() else "no",
            "Checkpoint Count": len(list(checkpoint_dir.glob("*.zip"))) if checkpoint_dir.exists() else 0,
        }

        metrics.update(_load_eval_metrics(eval_npz_path))

        if progress_path.exists():
            try:
                df = pd.read_csv(progress_path)
                if not df.empty:
                    if "time/total_timesteps" in df.columns:
                        metrics["Total Timesteps"] = _safe_last_int(df["time/total_timesteps"])
                    else:
                        metrics["Total Timesteps"] = "N/A"

                    metrics["Final Mean Reward"] = (
                        _safe_last(df["rollout/ep_rew_mean"], precision=2)
                        if "rollout/ep_rew_mean" in df.columns
                        else "N/A"
                    )
                    metrics["Final Value Loss"] = (
                        _safe_last(df["train/value_loss"], precision=4) if "train/value_loss" in df.columns else "N/A"
                    )
                    metrics["Final Entropy"] = (
                        _safe_last(df["train/entropy_loss"], precision=4)
                        if "train/entropy_loss" in df.columns
                        else "N/A"
                    )
                    metrics["Final Approx KL"] = (
                        _safe_last(df["train/approx_kl"], precision=6) if "train/approx_kl" in df.columns else "N/A"
                    )
            except Exception as e:
                print(f"Error reading {progress_path}: {e}")

        run_data.append(metrics)

    return run_data


def generate_html(data, output_file: Path):
    if not data:
        print("No data collected to generate report.")
        return None

    df = pd.DataFrame(data)

    if "Best Eval Mean Reward" in df.columns:
        sort_col = pd.to_numeric(df["Best Eval Mean Reward"], errors="coerce")
        if sort_col.notna().any():
            df = df.assign(_sort_col=sort_col).sort_values(by="_sort_col", ascending=False).drop(columns=["_sort_col"])
    elif "Final Mean Reward" in df.columns:
        sort_col = pd.to_numeric(df["Final Mean Reward"], errors="coerce")
        if sort_col.notna().any():
            df = df.assign(_sort_col=sort_col).sort_values(by="_sort_col", ascending=False).drop(columns=["_sort_col"])

    html_table = df.to_html(classes="compare-table", index=False, border=0)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RL Training Comparison Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0f172a;
                --card-bg: #1e293b;
                --text: #f8fafc;
                --accent: #38bdf8;
                --border: #334155;
            }}
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                width: 100%;
                max-width: 1400px;
            }}
            h1 {{
                font-weight: 600;
                font-size: 2.5rem;
                margin-bottom: 30px;
                background: linear-gradient(to right, #38bdf8, #818cf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .compare-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: var(--card-bg);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }}
            th {{
                background-color: var(--border);
                color: var(--accent);
                text-align: left;
                padding: 16px;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.8rem;
                letter-spacing: 0.05em;
                white-space: nowrap;
            }}
            td {{
                padding: 14px 16px;
                border-bottom: 1px solid var(--border);
                font-size: 0.92rem;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            tr:hover td {{
                background-color: rgba(56, 189, 248, 0.05);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Training Comparison</h1>
            {html_table}
        </div>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_template)

    print(f"\nReport generated at: {output_file.absolute()}")
    return output_file


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    RUNS_DIR = ROOT / "runs"
    OUTPUT_FILE = ROOT / "comparison_report.html"

    data = collect_run_data(RUNS_DIR)
    report_path = generate_html(data, OUTPUT_FILE)

    if report_path:
        try:
            webbrowser.open(f"file://{report_path.absolute()}")
        except Exception:
            pass
