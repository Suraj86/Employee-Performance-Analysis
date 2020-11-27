# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'SVM.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        EmpLastSalaryHikePercent = int(request.form['EmpLastSalaryHikePercent'])
        EmpEnvironmentSatisfaction = int(request.form['EmpEnvironmentSatisfaction'])
        YearsSinceLastPromotion = int(request.form['YearsSinceLastPromotion'])
        EmpDepartment = int(request.form['EmpDepartment'])
        ExperienceYearsInCurrentRole = int(request.form['ExperienceYearsInCurrentRole'])
        EmpWorkLifeBalance = float(request.form['EmpWorkLifeBalance'])
        YearsWithCurrManager = float(request.form['YearsWithCurrManager'])
        EmpJobRole = int(request.form['EmpJobRole'])
        ExperienceYearsAtThisCompany = int(request.form['ExperienceYearsAtThisCompany'])
        
        data = np.array([[EmpLastSalaryHikePercent, EmpEnvironmentSatisfaction,
                          YearsSinceLastPromotion, EmpDepartment,
                          ExperienceYearsInCurrentRole, EmpWorkLifeBalance,
                          YearsWithCurrManager, EmpJobRole,
                          ExperienceYearsAtThisCompany]])
        my_prediction = grid.predict(data)
        
        output = round(prediction[0], 3)
        
        return render_template('result.html', prediction_text='Employee Performance Rate is {}'.format(output))
if __name__ == '__main__':
	app.run(debug=True)