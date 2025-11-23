from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:
        # Collect NEW dataset inputs
        data = CustomData(
            date=request.form.get('date'),
            down_hole_presure=request.form.get('down_hole_presure'),
            down_hole_temperature=request.form.get('down_hole_temperature'),
            production_pipe_pressure=request.form.get('production_pipe_pressure'),
            choke_size_pct=request.form.get('choke_size_pct'),
            well_head_presure=request.form.get('well_head_presure'),
            well_head_temperature=request.form.get('well_head_temperature'),
            choke_size_pressure=request.form.get('choke_size_pressure'),
        )

        # Convert inputs → DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input dataframe:")
        print(pred_df)

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Return result to UI
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
