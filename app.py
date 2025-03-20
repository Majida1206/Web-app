import pickle
import numpy as np
from flask import Flask ,render_template,request

app=Flask(__name__)
model=pickle.load(open('lr_model.pickle','rb'))
@app.route('/') #Root url
def home():
    return render_template('wine.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
     volatile_acidity= float(request.form['volatile_acidity'])
     citric_acid=float(request.form['citric_acid'])
     residual_sugar=float(request.form['residual_sugar'])
     chlorides=float(request.form['chlorides'])
     total_sulfur_dioxide=float(request.form['total_sulfur_dioxide'])
     density=float(request.form['density'])
     pH=float(request.form['pH'])
     sulphates=float(request.form['sulphates'])
     alcohol=float(request.form['alcohol'])                 
     features = [float(request.form.get(feature)) for feature in [
            'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]]
     final_input = np.array([features])  # 2D array for prediction
     prediction = model.predict(final_input)
     predicted_quality = round(prediction[0], 2)
     result = "Purchased" if prediction[0] >=6 else "Not purchased"

     return render_template('purchaseresult.html', prediction_text= result)
    
    except Exception as e:
        return render_template('purchaseresult.html', prediction_text=f'Error occurred: {str(e)}')
     




if __name__ == '__main__':
    app.run(debug=True)