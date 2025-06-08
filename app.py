from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        prediction = model.predict([input_data])[0]
        return render_template('result.html', prediction=prediction)
    except:
        return render_template('result.html', prediction="Error in prediction")

if __name__ == '__main__':
    app.run(debug=True)
