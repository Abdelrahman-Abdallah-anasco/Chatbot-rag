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


# @app.route("/upload", methods=["POST"])
# def upload_docx():
#     """
#     Accept a single DOCX upload, save it, then add it to the RAG index.
#     """
#     if "file" not in request.files:
#         return jsonify(error="No file sent"), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify(error="Empty filename"), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(UPLOAD_DIR, filename)

#     # Save the file
#     with open(file_path, "wb") as f:
#         shutil.copyfileobj(file.stream, f)

#     # Pass it to your ingestion pipeline
#     try:
#         add_new_doc(file_path)
#     except Exception as exc:
#         return jsonify(error=str(exc)), 500

#     return jsonify(message="Document added successfully"), 201


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
        result = qa_chain.invoke({"query": data["question"]})
        sources = [os.path.basename(d.metadata.get("source", "Unknown")) for d in result["source_documents"]]
        return jsonify(answer=result["result"], sources=sources)
    except Exception as exc:
        import traceback
        traceback.print_exc()  # Print the full traceback to console
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    # You can change host/port or read from env vars as needed
    app.run(host="0.0.0.0", port=8000, debug=False)
