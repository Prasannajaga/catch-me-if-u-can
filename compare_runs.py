import pandas as pd
from pathlib import Path
import json
import webbrowser

def collect_run_data(runs_root: Path):
    run_data = []
    
    if not runs_root.exists():
        print(f"Directory {runs_root} does not exist.")
        return []

    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
            
        config_path = run_dir / "config.json"
        progress_path = run_dir / "progress.csv"
        
        if not config_path.exists():
            continue
            
        # Load Config
        with open(config_path, "r") as f:
            config = json.load(f)
            
        metrics = {
            "Run Name": config.get("run_name", run_dir.name),
            "Learning Rate": config.get("learning_rate", "N/A"),
            "Batch Size": config.get("batch_size", "N/A"),
            "Steps": config.get("n_steps", "N/A"),
            "Ent Coef": config.get("ent_coef", "N/A"),
            "Gamma": config.get("gamma", "N/A"),
        }
        
        # Load Progress Metrics
        if progress_path.exists():
            try:
                df = pd.read_csv(progress_path)
                if not df.empty:
                    metrics["Total Timesteps"] = f"{df['time/total_timesteps'].iloc[-1]:,}"
                    metrics["Final Mean Reward"] = round(df["rollout/ep_rew_mean"].iloc[-1], 2) if "rollout/ep_rew_mean" in df.columns else "N/A"
                    metrics["Final Value Loss"] = round(df["train/value_loss"].iloc[-1], 4) if "train/value_loss" in df.columns else "N/A"
                    metrics["Final Entropy"] = round(df["train/entropy_loss"].iloc[-1], 4) if "train/entropy_loss" in df.columns else "N/A"
                    metrics["Final Approx KL"] = round(df["train/approx_kl"].iloc[-1], 6) if "train/approx_kl" in df.columns else "N/A"
            except Exception as e:
                print(f"Error reading {progress_path}: {e}")
        
        run_data.append(metrics)
        
    return run_data

def generate_html(data, output_file: Path):
    if not data:
        print("No data collected to generate report.")
        return

    df = pd.DataFrame(data)
    
    # Sort by reward if available
    if "Final Mean Reward" in df.columns:
        df = df.sort_values(by="Final Mean Reward", ascending=False)

    html_table = df.to_html(classes='compare-table', index=False, border=0)
    
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
                max-width: 1200px;
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
            }}
            td {{
                padding: 16px;
                border-bottom: 1px solid var(--border);
                font-size: 0.95rem;
            }}
            tr:last-child td {{
                border-bottom: none;
            }}
            tr:hover td {{
                background-color: rgba(56, 189, 248, 0.05);
            }}
            .badge {{
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                background: var(--border);
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
        # Try to open in browser automatically
        try:
            webbrowser.open(f"file://{{report_path.absolute()}}")
        except:
            pass
