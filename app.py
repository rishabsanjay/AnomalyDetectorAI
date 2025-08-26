from __future__ import annotations
import os, uuid
from typing import List

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')  # headless for servers
import matplotlib.pyplot as plt

from model import AnomalyDetector
from data_gen import generate_synthetic

# ---- paths / config ----
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = os.path.join('static', 'plots')
ALLOWED_EXTENSIONS = {'.csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-change-me'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---- helpers ----
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def load_csv_numeric(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'time' in df.columns and 'time' not in numeric.columns:
        numeric.insert(0, 'time', df['time'].values)
    if 'time' not in numeric.columns:
        numeric.insert(0, 'time', np.arange(len(numeric)))
    return numeric


def fit_and_score(df_num: pd.DataFrame, feature_cols: List[str] | None = None, contamination: float = 0.05):
    if feature_cols is None:
        feature_cols = [c for c in df_num.columns if c != 'time']
    if not feature_cols:
        raise ValueError("No numeric feature columns found (need at least one besides 'time').")

    X = df_num[feature_cols].values
    split = max(10, int(0.6 * len(X)))  # assume first 60% is mostly normal
    X_train = X[:split]

    detector = AnomalyDetector(contamination=contamination)
    detector.fit(X_train)
    scores, labels = detector.predict(X)

    out = df_num.copy()
    out['anomaly_score'] = scores
    out['is_anomaly'] = labels
    return out, feature_cols


def plot_series(df_scored: pd.DataFrame, feature_cols: List[str]) -> str:
    """Neon-styled plot for Cyberpunk theme."""
    fname = f"{uuid.uuid4().hex}.png"
    fpath = os.path.join(PLOT_FOLDER, fname)

    plt.rcParams.update({
        "axes.edgecolor": "#1f2a44",
        "axes.labelcolor": "#eaf2ff",
        "xtick.color": "#c7d2fe",
        "ytick.color": "#c7d2fe",
        "grid.color": "#1f2a44",
        "figure.facecolor": (0, 0, 0, 0),
        "axes.facecolor": (0, 0, 0, 0),
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    x = df_scored['time'].values
    y = df_scored[feature_cols[0]].values

    # Neon line
    ax.plot(x, y, linewidth=1.6, color="#60a5fa")
    ax.grid(True, alpha=0.25, linestyle="--")

    # Neon anomaly points
    anom = df_scored[df_scored['is_anomaly'] == 1]
    if not anom.empty:
        ax.scatter(
            anom['time'].values,
            anom[feature_cols[0]].values,
            s=26,
            color="#22d3ee",         # cyan
            edgecolors="#f472b6",    # pink edge
            linewidths=0.6,
            zorder=5,
        )

    ax.set_title(f"{feature_cols[0]} — anomalies highlighted", color="#eaf2ff", fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel(feature_cols[0])

    plt.tight_layout()
    fig.savefig(fpath, dpi=160, transparent=True)
    plt.close(fig)
    return os.path.join('static', 'plots', fname)


# ---- routes ----
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('No file part in request')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if not (file and allowed_file(file.filename)):
        flash('Unsupported file type. Please upload a CSV file.')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(fpath)

    df = load_csv_numeric(fpath)
    contamination = float(request.form.get('contamination', '0.05'))
    topk = int(request.form.get('topk', '30'))

    df_scored, feats = fit_and_score(df, contamination=contamination)
    plot_path = plot_series(df_scored, feats)

    top = df_scored.sort_values('anomaly_score', ascending=False).head(topk)
    cols = ['time'] + feats + ['anomaly_score', 'is_anomaly']
    table = top[cols].round(4).to_dict(orient='records')

    return render_template(
        'results.html',
        plot_path=plot_path,
        rows=table,
        columns=cols,
        total=len(df_scored),
        anomalies=int(df_scored['is_anomaly'].sum()),
        contamination=contamination,
        features=', '.join(feats),
    )


@app.route('/demo', methods=['GET'])
def demo():
    df = generate_synthetic()
    tmp_name = f"demo_{uuid.uuid4().hex}.csv"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], tmp_name)
    df.to_csv(fpath, index=False)

    df_num = df.select_dtypes(include=[np.number]).copy()
    if 'time' not in df_num.columns and 'time' in df.columns:
        df_num.insert(0, 'time', df['time'].values)

    df_scored, feats = fit_and_score(df_num)
    plot_path = plot_series(df_scored, feats)

    top = df_scored.sort_values('anomaly_score', ascending=False).head(30)
    cols = ['time'] + feats + ['anomaly_score', 'is_anomaly']
    table = top[cols].round(4).to_dict(orient='records')

    return render_template(
        'results.html',
        plot_path=plot_path,
        rows=table,
        columns=cols,
        total=len(df_scored),
        anomalies=int(df_scored['is_anomaly'].sum()),
        contamination=0.05,
        features=', '.join(feats),
    )


if __name__ == '__main__':
    app.run(debug=True)

