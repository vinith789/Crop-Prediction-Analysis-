from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=["POST"])
def prediction():
    try:
        # Get input values from the form
        temp = float(request.form.get("temp"))
        humd = float(request.form.get("humd"))
        ph = float(request.form.get("ph"))
        rain = float(request.form.get("rain"))

        # Load the dataset
        df = pd.read_csv("crop_dataset.csv")

        # Prepare the data
        y_train = df["label"]
        x_train = df.drop(columns=["label"])

        # Train the KNeighborsClassifier model
        clf_knn = KNeighborsClassifier(n_neighbors=3)
        clf_knn.fit(x_train, y_train)

        # Predict the crop
        x_test = [[temp, humd, ph, rain]]
        pre_res = clf_knn.predict(x_test)

        return render_template('result.html', prediction=pre_res[0])
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host="localhost", port=1166, debug=True)
