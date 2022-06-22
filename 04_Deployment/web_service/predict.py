import pickle
from flask import Flask, request, jsonify

with open('models/model.bin', 'rb') as f_in:
    model = pickle.load(f_in)
with open('models/dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

def predict(ride):
    X = dv.transform(ride)
    preds = model.predict(X)
    return preds[0]


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    pred = predict(ride)

    result = {"duration":pred}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)