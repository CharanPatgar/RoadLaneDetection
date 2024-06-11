from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_lane', methods=['POST'])
def detect_lane():
    if request.method == 'POST':
        # Execute ln.py
        subprocess.run(['python', 'ln.py'])
        return 'Lane detection completed.'

if __name__ == '__main__':
    app.run(debug=True)
