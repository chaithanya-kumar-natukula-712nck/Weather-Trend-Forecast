# 🌦️ Weather Trend Forecasting using Time Series Analysis

## 📌 Project Overview
This project focuses on analyzing and forecasting weather trends using historical meteorological data.  
The primary goal is to understand temperature patterns over time and predict future trends using time series forecasting techniques.

The project demonstrates strong data science fundamentals including data preprocessing, exploratory data analysis (EDA), time series modeling, and evaluation using multiple metrics.

---

## 🎯 Objectives
- Analyze historical weather data to identify trends and seasonality
- Perform data cleaning and preprocessing
- Build time series forecasting models using **SARIMA** and **Prophet**
- Compare model performance using standard evaluation metrics
- Improve prediction accuracy using an **ensemble approach**
- Forecast future temperature trends for a selected region (India)

---

## 📂 Dataset
- **Source:** Global Weather Repository (Kaggle)
- **Description:** Daily weather information for cities across the world
- **Key Features Used:**
  - `temperature_celsius`
  - `last_updated`
  - Country and location-based attributes

> For time series analysis, the dataset was filtered to include **India only**, and the `last_updated` column was used as the time index.

---

## 🛠️ Technologies & Tools
- **Programming Language:** Python  
- **Libraries Used:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Statsmodels
  - Prophet

---

## 🧹 Data Preprocessing
The following preprocessing steps were applied:

- Handling missing values
- Removing or capping outliers using **IQR** and **Z-score**
- Converting `last_updated` to datetime format
- Sorting data chronologically
- Setting `last_updated` as the time series index
- No normalization was applied, as it may reduce interpretability and forecasting accuracy for SARIMA

---

## 📊 Exploratory Data Analysis (EDA)
EDA was conducted to understand the underlying structure of the data:

- Distribution of temperature values
- Detection of outliers using boxplots
- Trend visualization over time
- Seasonal patterns and fluctuations
- Country-wise and location-wise comparisons (before filtering to India)

---

## ⏳ Time Series Analysis
Key components analyzed:
- **Trend:** Long-term movement in temperature
- **Seasonality:** Repeating yearly patterns
- **Noise:** Random variations

Rolling statistics and decomposition techniques were used to better understand the time series behavior.

---

## 🤖 Models Used

### 1️⃣ SARIMA (Seasonal ARIMA)
- Captures both trend and seasonal components
- Well-suited for short-term forecasting
- Parameters tuned based on ACF and PACF analysis

### 2️⃣ Prophet
- Designed to handle strong seasonality and missing data
- More effective for long-term forecasting
- Automatically detects changepoints

### 3️⃣ Ensemble Model
- Combined predictions from SARIMA and Prophet
- Helps reduce model-specific bias
- Improved overall forecast stability and accuracy

---

## 📈 Model Evaluation
The models were evaluated using the following metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**

Lower values indicate better performance.  
The ensemble approach showed more balanced and reliable results compared to individual models.

---

## 🔮 Forecasting Results
- SARIMA performed well in capturing short-term temperature fluctuations
- Prophet handled long-term seasonal trends effectively
- Ensemble model provided smoother and more accurate predictions
- Forecast plots clearly show expected future temperature trends

---

## 📌 Key Insights
- Temperature exhibits clear seasonal behavior
- Short-term forecasting benefits from SARIMA
- Long-term forecasting benefits from Prophet
- Ensemble methods improve robustness and accuracy
- Proper preprocessing significantly impacts model performance

---

## 📁 Project Structure
```
├── data/
│ └── GlobalWeatherRepository.csv
├── notebooks/
│ └── Weather_Trend_Forecasting.ipynb
├── images/
│ └── plots and visualizations
├── README.md
```

---

## 🚀 Future Improvements
- Include additional weather parameters (humidity, rainfall, wind speed)
- Apply deep learning models such as LSTM or GRU
- Automate hyperparameter tuning
- Extend forecasting to multiple countries
- Deploy the model using a web application or API

---

## 🧾 Conclusion
This project demonstrates the effectiveness of time series analysis in understanding and predicting weather patterns.  
By leveraging SARIMA, Prophet, and an ensemble approach, the study successfully forecasts temperature trends with reliable accuracy.  
The project highlights the importance of preprocessing, model selection, and evaluation in real-world forecasting problems.

## Installation and Usage

To run the project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/chaithanya-kumar-natukula-712nck/Weather-Trend-Forecast.git
cd Weather-Trend-Forecast

# Install required dependencies
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels prophet
```

## Run the Project

This project is implemented using a Jupyter Notebook.

After installing the dependencies, launch Jupyter Notebook:

```bash
jupyter notebook
```
Open the following notebook and run all cells sequentially:
```
Weather_Trend_Forecasting.ipynb
```
---
## Acknowledgments
Special thanks to **Kaggle** for providing the dataset and to the open-source community for developing powerful forecasting libraries such as **Prophet** and **Statsmodels**.

---

## 📬 Contact
**Chaithanya**  
📧 Email: your-email@gmail.com  
🔗 GitHub: https://github.com/chaithanya-kumar-natukula-712nck  
🔗 LinkedIn: https://www.linkedin.com/in/chaithanya-kumar-natukula-712nck

---

## 📜 License
This project is open-source and available under the **MIT License**.
