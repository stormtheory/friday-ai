from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"  # Change as needed

### Command to interface with the api helper
# curl -X POST http://localhost:5000/api/chat \
#  -H "Content-Type: application/json" \
#  -d '{"prompt": "What is the capital of France?"}'


@app.route('/api/chat', methods=['POST'])
def chat_with_ollama():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        model = data.get("model", DEFAULT_MODEL)

        # Send the prompt to Ollama's API
        ollama_response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False  # Set to True if you want streaming
        })

        # Forward the response back to the client
        if ollama_response.ok:
            return jsonify(ollama_response.json()), 200
        else:
            return jsonify({"error": "Failed to query Ollama"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run Flask app on localhost, not exposed externally by default
    app.run(host='127.0.0.1', port=5000, debug=True)
