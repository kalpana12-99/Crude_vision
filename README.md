# 🛢️ Volve Field Oil Production Prediction  
A complete end-to-end Machine Learning + Flask web application that predicts **oil production** using real wellhead and downhole sensor data from the Volve oil field.

This project demonstrates how ML can support petroleum engineering operations by predicting oil output in real time, enabling better decision-making and optimization.

---

# 🎯 Why This Project Was Created

In the petroleum industry, wells generate huge volumes of sensor data every second — pressures, temperatures, choke settings, and fluid measurements.  
However, **accurately predicting oil production instantly** is still challenging due to:

- Complex reservoir behavior  
- Non-linear relationships between variables  
- Rapid changes in well conditions  
- Operational uncertainties  
- Limited availability of real-time predictive tools  

This project was built to show how Machine Learning can transform these raw signals into **actionable predictions** that help petroleum engineers optimize production.

---

# 🛢️ How This Project Helps the Petroleum Industry

### ✅ 1. Real-Time Production Prediction  
The ML model predicts oil output using only sensor inputs, helping engineers monitor well performance instantly.

### ✅ 2. Early Anomaly Detection  
Sudden deviations in predicted values may indicate:

- Water breakthrough  
- Gas coning  
- Scale buildup  
- Artificial lift issues  
- Reservoir flow restrictions  

Thus, the system supports **proactive well intervention**.

### ✅ 3. Choke & Pressure Optimization  
Engineers can experiment with choke settings, pressures, or temperature changes and immediately see predicted impacts.

### ✅ 4. Faster than Reservoir Simulators  
Full-physics simulators are slow.  
This ML model provides **quick estimations** that support real-time decisions.

### ✅ 5. Useful for Digital Oilfield Automation  
This pipeline can be integrated into:

- SCADA  
- Digital twin systems  
- Real-time dashboards  
- Production optimization software  

Making the wellsite more intelligent and automated.

### ✅ 6. Complete ML Workflow for Oilfield Data  
This project includes everything:

- Data ingestion  
- Processing  
- Feature engineering  
- Model training  
- Pipeline saving  
- Flask UI for prediction  

It’s a perfect demonstration of how ML can be applied in petroleum engineering.

---

# 🚀 Project Overview

The model predicts **oil production (`oil`)** based on the following features:

- `date`
- `down_hole_presure`
- `down_hole_temperature`
- `production_pipe_pressure`
- `choke_size_pct`
- `well_head_presure`
- `well_head_temperature`
- `choke_size_pressure`

Target:
- **`oil`**

---

# 🧠 Machine Learning Pipeline

## ✔ 1. Data Ingestion  
- Reads `volve_field_data.csv`  
- Converts date to datetime  
- Stores:
  - `artifacts/data.csv`
  - `artifacts/train.csv`
  - `artifacts/test.csv`

## ✔ 2. Data Transformation  
- Extracts `month` and `day_of_year`  
- Scales all numeric features  
- Saves `preprocessor.pkl`

## ✔ 3. Model Training  
Trains multiple models:

- RandomForestRegressor  
- DecisionTreeRegressor  
- GradientBoostingRegressor  
- LinearRegression  
- XGBRegressor  
- CatBoostRegressor  
- AdaBoostRegressor  

Best model saved as:
```
artifacts/model.pkl
```

---

# 🌐 Flask Web Application

## `index.html`
- Welcome page  
- Button to access the prediction form  

## `home.html`
User inputs:

- Date  
- Down-hole pressure  
- Down-hole temperature  
- Production pipe pressure  
- Choke size (%)  
- Well head pressure  
- Well head temperature  
- Choke size pressure  

The app uses:
- `preprocessor.pkl`  
- `model.pkl`  
to produce a real-time prediction.

---

# 🗂 Project Structure

```
CRUDE VISION/
│── app.py
│── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│── templates/
│   ├── index.html
│   └── home.html
│── static/
│── requirements.txt
│── README.md
```

---

# ▶️ How to Run the Project

## 1️⃣ Create & activate virtual environment
```bash
python -m venv env
env\Scripts\activate   # Windows
```

## 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ (Optional) Train the pipeline
```bash
python -m src.components.data_ingestion
```

## 4️⃣ Run the Flask app
```bash
python app.py
```

Visit:
👉 **http://127.0.0.1:5000/**

---

# 🛠 Technology Stack

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **XGBoost**
- **CatBoost**
- **Flask**
- **HTML/CSS**

---

# 📈 Future Improvements

- Deploy on Render / AWS / Railway  
- Add sensor dashboards  
- Build time-series LSTM model  
- Add anomaly detection  
- Integrate with programmable choke systems  

---

# 🙌 Acknowledgements

Dataset inspired by the **Volve Field** public dataset.  
This project was built to demonstrate the integration of petroleum engineering with modern AI/ML techniques.

---
