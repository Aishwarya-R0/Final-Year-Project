from flask import Flask, render_template, request, redirect, url_for, session
import sound, tocheck
app = Flask(__name__, template_folder='templates')
@app.route('/', methods =['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/record', methods =['POST', 'GET'])
def record():
    sound.main()
    #return redirect("http://127.0.0.1:5005")
    return redirect('http://192.168.1.14:5005/record', code=302)

@app.route('/tester', methods =['POST', 'GET'])
def tester():
    val=tocheck.main()
    if val=='Depression':
        val='<center><h1>You seem to be under depression. Please do consult a doctor at the earliest</h1></center>'

    else:
        val='<center><h1>Hurray!! You are mentally healthy</h1></center>'
    return val
if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0', port=5000)
