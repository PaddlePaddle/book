from flask import Flask, jsonify, render_template, request
import os

# webapp
app = Flask(__name__)


@app.route('/')
def main():
    backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000/')
    return render_template('index.html', backend_url=backend_url)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
