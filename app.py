# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('SVM.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = grid.predict(final_features)
        
    output = round(prediction[0], 9)
        
    return render_template('result.html', prediction_text='Employee Performance Rate is {}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)