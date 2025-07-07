# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# your own helpers
from rag_engine import get_qa_chain

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def health_check():
    """Quick liveness probe."""
    return jsonify(status="ok")


@app.route("/query", methods=["POST"])
def query_doc():
    """
    Only acceept questions related to the trained documents.
    Run a user question through the QA chain and return answer + source filenames.
    Expected JSON payload: {"question": "<your question>"}
    """
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify(error="Payload must include 'question'"), 400

    try:
        qa_chain = get_qa_chain()
        result = qa_chain({"query": data["question"]})
        sources = [os.path.basename(d.metadata["source"]) for d in result["source_documents"]]

        return jsonify(answer=result["result"], sources=sources)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    # You can change host/port or read from env vars as needed
    app.run(host="0.0.0.0", port=8000, debug=False)
