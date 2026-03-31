# app.py

from flask import Flask, request, jsonify, session, send_from_directory
from agent import TravelMindAgent
import uuid

app = Flask(__name__, static_folder="static")
app.secret_key = "travelmind-secret-key"

_agents: dict[str, TravelMindAgent] = {}

def get_agent() -> TravelMindAgent:
    sid = session.get("sid")
    if not sid or sid not in _agents:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _agents[sid] = TravelMindAgent()
    return _agents[sid]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = (data.get("message") or "").strip()
        if not user_input:
            return jsonify({"error": "Empty message."}), 400
        response = get_agent().chat(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    try:
        get_agent().reset()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)