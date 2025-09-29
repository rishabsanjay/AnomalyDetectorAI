AI Anomaly Detector

AI Anomaly Detector is a Flask-based web application that uses machine learning to automatically identify unusual patterns in test data. Engineers can upload sensor datasets (like rocket engine telemetry or vehicle diagnostics), and the app highlights potential anomalies with interactive neon-styled visualizations inspired by a Cyberpunk Mission Control dashboard.

Features

- Upload CSVs — Analyze your own sensor/test datasets.

- AI Model — Isolation Forest detects anomalies in numeric columns.

- Neon Visualization — Anomalies are flagged on glowing cyberpunk plots.

- Demo Mode — Try with a synthetic dataset if you don’t have your own.

- Cyberpunk UI — Glassmorphism cards, animated gradients, neon buttons.


How It Works

Upload Data — A CSV file with numeric sensor columns (and optional time column).

Model Training — Isolation Forest learns the baseline from the first portion of data.

Scoring — Each row gets an anomaly score.

Results — Anomalies are highlighted in a neon plot and listed in a sortable table.

Tech Stack

Backend: Flask (Python)

ML: scikit-learn (Isolation Forest)

Data: pandas, numpy

Visualization: matplotlib (neon custom theme)

Frontend: HTML + CSS (Orbitron/Rajdhani fonts, glassmorphism cyberpunk theme)

Getting Started
1. Clone the repo
git clone https://github.com/rishabsanjay/AnomalyDetectorAI.git
cd AnomalyDetectorAI

2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
python app.py


Then open http://127.0.0.1:5000
 in your browser.

Project Structure
AnomalyDetectorAI/
├── app.py               # Flask app entrypoint
├── model.py             # Isolation Forest wrapper
├── data_gen.py          # Synthetic demo data generator
├── requirements.txt     # Python dependencies
├── static/
│   ├── styles.css       # Cyberpunk theme CSS
│   └── plots/           # Saved plot images
├── templates/
│   ├── base.html        # Layout
│   ├── index.html       # Upload form
│   └── results.html     # Results page
└── uploads/             # Uploaded CSV files

Demo Dataset

Don’t have data handy? Click Demo Dataset in the app to generate synthetic sensor data with injected anomalies.
