# Minimal Flask app for Render deployment
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load predictions
pred_df = pd.read_csv("data/college_football_schedule_2025_predicted_totals_enhanced.csv")
team_conf_df = pd.read_csv("data/team_conferences.csv")

@app.route('/')
def index():
    sample = pred_df.head(10)
    return sample.to_html()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
