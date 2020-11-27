import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'EmpLastSalaryHikePercent':13,'EmpEnvironmentSatisfaction':2,
                            'YearsSinceLastPromotion':1,'EmpDepartment':1,
                            'ExperienceYearsInCurrentRole':4,
                            'EmpWorkLifeBalance':4,'YearsWithCurrManager':5,
                             'EmpJobRole':1,'ExperienceYearsAtThisCompany':6})

print(r.json())