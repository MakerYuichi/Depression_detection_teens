from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras

app = Flask(__name__, template_folder='templates')

# Load the trained model with compile=False to avoid optimizer issues
model = keras.models.load_model('ann_model.keras', compile=False)

@app.route('/')
def welcome():
    return render_template("welcome.html")

@app.route('/dep_test', methods=['POST', 'GET'])
def dep_test():
    if request.method == 'POST':
        try:
            # Convert inputs to float (instead of int) to avoid errors
            values = [float(x) for x in request.form.values()]
            final = np.array(values).reshape(1, -1)

            # Ensure input shape matches model expectation
            if final.shape[1] != model.input_shape[1]:
                return render_template("result.html", results="Invalid input dimensions.")

            # Predict class
            prediction = model.predict(final)
            int_predict = np.argmax(prediction)  # Extract class index

            # Mapping numerical prediction to category
            label_map = {0: "High", 1: "Low", 2: "Mild", 3: "Moderate"}
            result = label_map.get(int_predict, "Unknown")

            return render_template("result.html", results=result)
        except Exception as e:
            return render_template("result.html", results=f"Error: {str(e)}")
    else:
        return render_template('dep_test.html')

if __name__ == "__main__":
    app.run(debug=True)
