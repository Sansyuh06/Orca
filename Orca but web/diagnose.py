import os
import sys

print("--- DIAGNOSTIC START ---")
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")
print("Directory Contents:")
try:
    for item in os.listdir('.'):
        print(f" - {item}")
except Exception as e:
    print(f"Error listing directory: {e}")

print("--- DIAGNOSTIC END ---")
# Keep running so container doesn't exit immediately (which counts as crash)
from flask import Flask
app = Flask(__name__)
@app.route('/')
def index(): return "Diagnostic Running"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
