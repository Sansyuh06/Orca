import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from Minimal App! Deployment Works!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting minimal app on port {port}")
    app.run(host='0.0.0.0', port=port)
