import os
import pandas as pd
from flask import Flask, jsonify, render_template
from openai import OpenAI

# ---------- Config ----------
CSV_PATH = "stock.csv"  # path to your stock file
client = OpenAI()

app = Flask(__name__, template_folder="templates", static_folder="static")


def load_stock_csv_as_text() -> str:
    """
    Load the CSV and return it as a CSV string the model can see.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    # You can print a preview in the server logs if you want
    print("=== Stock preview ===")
    print(df.head())
    print("=====================")
    return df.to_csv(index=False)


@app.route("/")
def index():
    # Just serve the HTML page
    return render_template("index.html")


@app.route("/client-secret")
def client_secret():
    """
    Returns a short-lived Realtime client secret to the browser.
    The CSV is embedded in the session instructions so the voice agent
    can answer questions based ONLY on this stock data.
    """
    csv_text = load_stock_csv_as_text()

    instructions = f"""
You are a friendly retail stock assistant.

You have access to the following stock table in CSV format:

{csv_text}

Rules:
- Answer questions ONLY using this CSV.
- If a product or detail is not in the CSV, say you don't know.
- Use exact product names from the CSV when possible.
- Be brief, clear and conversational.
"""

    secret = client.realtime.client_secrets.create(
        session={
            "type": "realtime",
            "model": "gpt-realtime",
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy",
                    "speed": 1.0,
                },
            },
            # <-- only ONE modality
            "output_modalities": ["audio"],
        }
    )


    # We only return the secret value; the browser never sees your real API key
    return jsonify({"client_secret": secret.value})


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
    app.run(debug=True, host="0.0.0.0", port=5000)
