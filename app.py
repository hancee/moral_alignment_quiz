from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

#Load data
#seed = 19
#x_train = pickle.load(open('x_train.pkl','rb'))
#y_train = pickle.load(open('y_train.pkl','rb'))

#Train model
#clf = XGBClassifier(random_state=seed, max_depth=2**3)
#_ = clf.fit(x_train, y_train)

#Load model
#clf = XGBClassifier()
#clf.load_model('moral_alignment_model.json')
clf = xgb.Booster({'nthread':4})
clf.load_model('moral_alignment_model.model')
classes = ['Chaotic Evil', 'Chaotic Good', 'Chaotic Neutral',
           'Lawful Evil', 'Lawful Good', 'Lawful Neutral',
           'Neutral Evil', 'Neutral Good', 'True Neutral']

#Instantiate app
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    #Get responses
    responses = {}
    #for key in request.form:
    #    print(key, type(key), request.form[key], type(request.form[key]))
    #    responses[key] = [float(request.form[key])]
    keys = [key for key in request.form]
    for i in range(len(keys)):
        responses['f%d' % i] = [float(request.form[keys[i]])]
    input_df = pd.DataFrame.from_dict(responses)
    #pred = clf.predict(input_df)
    prob = clf.predict(xgb.DMatrix(input_df))[0].tolist()
    pred = classes[prob.index(max(prob))]
    print(pred)

    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
