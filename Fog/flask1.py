from flask import Flask, render_template, request, redirect, url_for, session
import test
app = Flask(__name__, template_folder='templates')
@app.route('/', methods =['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/record', methods =['POST', 'GET'])
def record():
    val=test.main()
    if val[1]=='0':
    	val='<center><h1>You seem to be under depression. Please do consult a doctor at the earliest</h1></center>'
    else:
    	val='<center><h1>Hurray!! You are mentally healthy</h1></center>'

    return str(val)
if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0', port=5005)
