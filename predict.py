from flask import Flask, request, jsonify

import pickle

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform(customer)
    y_pred = model.predict(X)

    if y_pred[0] == 0:
        result = 'No'
    else:
        result = 'Yes'

    return {
        'churn': result
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)