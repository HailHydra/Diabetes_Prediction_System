from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_datapoint():
    # Collecting form data from the front-end
    data = CustomData(
        Pregnancies=int(request.form.get("Pregnancies")),
        Glucose=float(request.form.get("Glucose")),
        BloodPressure=float(request.form.get("BloodPressure")),
        SkinThickness=float(request.form.get("SkinThickness")),
        Insulin=float(request.form.get("Insulin")),
        BMI=float(request.form.get("BMI")),
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction")),
        Age=int(request.form.get("Age"))
    )

    # Converting input data to a dataframe
    final_data = data.get_data_as_dataframe()

    # Creating an instance of the prediction pipeline
    predict_pipeline = PredictPipeline()

    # Getting the prediction for the data
    pred = predict_pipeline.predict(final_data)

    # Rounding off the prediction result
    result = round(pred[0], 2)

    if result == 1:
        final_result1 = "The assessment indicates a higher likelihood of developing diabetes in the future."
    else:
        final_result1 = "The assessment indicates a lower likelihood of developing diabetes in the future."

    # Rendering the result on the HTML page
    return render_template("index.html", final_result=final_result1)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)