from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# model = None
with open('./rentcar_app/model.pkl', 'rb') as pickle_file:
    model= pickle.load(pickle_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def pred():
    data1 = request.form['차종']
    data2 = request.form['연료']
    data3 = request.form['탑승 인원']
    data4 = request.form['국산/외 분류']
    data5 = request.form['렌트 이용 계절'] 

    arr = [[data1, data2, int(data3), data4, data5], [data1, data2, int(data3), data4, data5]]

    df= pd.DataFrame(arr, columns=['VHCLE_TY_NM', 'VHCLE_FUEL_NM', 'VHCLE_NMPR_CO', 'VHCLE_MAKR_NM', 'USE_DT'])
    print(df)
    y_pred = model.predict(df[0:1])
    return render_template('result.html', data= y_pred)

if __name__ == "__main__":
    app.run(debug=True)
