from flask import Flask,request,jsonify, render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello"

@app.route('/predict',methods=['GET'])
def predict():
    args=request.args
    cgpa = float(args.get("cgpa"))
    iq = int(args.get("iq"))
    profile_score = int(args.get("profile_score"))

    input_query = np.array([[cgpa,iq,profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run()