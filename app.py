from flask import Flask, render_template, request
import numpy as np
import joblib


print("Starting Flask app...")

app = Flask(__name__, template_folder='templates')
model = joblib.load('model.pkl')
print(type(model))

@app.route('/')
def welcome():
    return render_template("welcome.html")


@app.route('/dep_test', methods=['POST','GET'])
def dep_test():
    if request.method == 'POST':
        values = [int(x) for x in request.form.values()]
        final = np.array(values).reshape(1,-1)
        int_predict = model.predict(final)
        if int_predict == 0:
            int_predict = "High"
        elif int_predict == 1:
            int_predict = "Low"
        elif int_predict == 2:
            int_predict = "Mild"
        else:
            int_predict = "Moderate"

        return render_template("result.html", results=int_predict)
    else:
        return render_template('dep_test.html')



if __name__ == "__main__":
    app.run(debug=True)



