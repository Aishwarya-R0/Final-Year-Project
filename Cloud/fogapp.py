from flask import Flask, render_template, request, redirect, url_for, session
import re, test
app = Flask(__name__, template_folder='templates')
@app.route('/', methods =['POST', 'GET'])
def home():
    return render_template('index1.html')


@app.route('/record', methods =['POST', 'GET'])
def record():
    return redirect('http://3.15.200.244:5005/index', code=302)
if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0', port=5000)