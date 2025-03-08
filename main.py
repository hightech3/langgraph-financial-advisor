import os
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from agent import run_financial_advisor

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api", methods=["GET"])
def test():
    return jsonify({"message": "Hello, World!"})

@app.route("/api/chat/financial-advice", methods=["POST"])
def get_financial_advice():
    data = request.json
    query = data.get("query", "")
    
    def generate():
        for chunk in run_financial_advisor(query):
            yield chunk
    
    return Response(generate(), mimetype="text/markdown", headers={
        'X-Accel-Buffering': 'no',  # Disable Nginx buffering
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/markdown'
    })

if __name__ == "__main__":
    # Fix for Windows socket error
    import sys
    if sys.platform == 'win32':
        import os
        # Set Flask environment variables
        os.environ['FLASK_APP'] = 'main.py'
        os.environ['FLASK_ENV'] = 'development'
        # Use the Flask CLI to run the app
        from flask.cli import main
        sys.argv = ['flask', 'run', '--host=127.0.0.1', '--port=8080']
        main()
    else:
        # For non-Windows platforms, use the regular method
        app.run(host="0.0.0.0", port=8080, debug=True)