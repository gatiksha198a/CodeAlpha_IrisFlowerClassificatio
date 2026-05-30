# 🌸 Iris Flower Classification
### CodeAlpha Data Science Internship | Task 1

<p align="center">
  <img src="screenshots/home.png" width="800" alt="Iris App Home"/>
</p>

## 📌 Overview
An interactive **Streamlit web application** that classifies Iris flower species 
(Setosa, Versicolor, Virginica) using Machine Learning, based on sepal and petal measurements.

## 🎯 Features
- 🔮 **Live Prediction** — Adjust sliders and classify in real-time
- 📊 **EDA Dashboard** — Feature distributions, pair plots, correlation heatmap
- 📈 **Model Comparison** — KNN vs Random Forest with confusion matrices
- 🏆 **Feature Importance** — Visual ranking of features

## 🖥️ App Screenshots

### 🔮 Live Prediction
<p align="center">
  <img src="screenshots/prediction.png" width="800" alt="Prediction Screen"/>
</p>

### 📊 Exploratory Data Analysis
<p align="center">
  <img src="screenshots/eda.png" width="800" alt="EDA Tab"/>
</p>

### 📈 Model Performance
<p align="center">
  <img src="screenshots/model_performance.png" width="800" alt="Model Performance"/>
</p>

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.12 | Core Language |
| Streamlit | Web App Framework |
| Scikit-learn | ML Models (KNN, Random Forest) |
| Pandas & NumPy | Data Processing |
| Matplotlib & Seaborn | Visualizations |

## 🤖 Models Used
| Model | Accuracy |
|-------|----------|
| Random Forest | ~97% |
| K-Nearest Neighbors | ~97% |

## 🚀 How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/gatiksha198a/CodeAlpha_IrisFlowerClassification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run task1_iris_app.py
```

## 📁 Project Structure
```
├── task1_iris_app.py              # Streamlit app
├── Task1_Iris_Classification.py   # Standalone ML script
├── requirements.txt               # Dependencies
├── screenshots/                   # App screenshots
└── README.md
```

## 💡 Key Insights
- **Petal dimensions** (length & width) are the most important features for classification
- **Random Forest** and **KNN** both achieve ~97% accuracy on this dataset
- The dataset is perfectly balanced — 50 samples per species

## 👩‍💻 About
**Gatiksha** | Data Science Intern @ CodeAlpha  
Student ID: CA/DF1/101914 | Punjabi University, Patiala  
GitHub: [@gatiksha198a](https://github.com/gatiksha198a)

---
*Built with ❤️ as part of CodeAlpha Data Science Internship*
