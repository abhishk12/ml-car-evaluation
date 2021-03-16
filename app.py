import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("carModel.pickle", 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    ans = "NA"
    if prediction[0] == 0:
        ans = "acceptable"
    elif prediction[0] == 1:
        ans = "good"
    elif prediction[0] == 2:
        ans = "unacceptable"
    elif prediction[0] == 3:
        ans = "very good"

    return render_template('index.html', prediction_text='Class of car should be {}'.format(ans))


if __name__ == "__main__":
    app.run(debug=True)
