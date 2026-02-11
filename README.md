# 📊 Student Performance Predictor

A machine learning project that predicts student math scores based on various demographic and academic factors using multiple regression algorithms.

## 🎯 Overview

This project analyzes student performance data and builds predictive models to estimate math scores based on features like gender, race/ethnicity, parental education level, lunch type, test preparation course completion, and scores in reading and writing.

The project implements a complete ML pipeline including data ingestion, transformation, model training with hyperparameter tuning, and a web interface for predictions.

## ✨ Features

- **Multiple ML Models**: Implements and compares 11 different regression algorithms
- **Automated Model Selection**: Uses GridSearchCV for hyperparameter tuning
- **Data Pipeline**: Modular pipeline for data ingestion, transformation, and model training
- **Web Interface**: FastAPI-based web application for making predictions
- **Comprehensive Logging**: Custom logging system for tracking pipeline execution
- **Exception Handling**: Custom exception handling for better error tracking

## 🔧 Technologies Used

- **Python 3.13**
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: seaborn, matplotlib
- **Web Framework**: FastAPI
- **Template Engine**: Jinja2

## 📁 Project Structure

```
student-performance/
│
├── artifacts/              # Stores trained models and preprocessors
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train_data.csv
│   ├── test_data.csv
│   └── raw_data.csv
│
├── notebooks/              # Jupyter notebooks for EDA and experimentation
│   └── model_train.ipynb
│
├── src/                    # Source code
│   ├── components/         # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transform.py
│   │   └── model_trainer.py
│   ├── pipelines/          # Prediction pipelines
│   │   └── predict_pipeline.py
│   ├── exceptions.py       # Custom exception handling
│   ├── logger.py          # Logging configuration
│   └── utilities.py       # Helper functions
│
├── templates/              # HTML templates for web interface
├── app.py                 # FastAPI application
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup file
└── README.md             # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saikatcodec/student-performance.git
   cd student-performance
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Model

Run the data ingestion and model training pipeline:

```bash
python src/components/data_ingestion.py
```

This will:
- Load and split the dataset
- Transform the data using preprocessing pipelines
- Train and evaluate multiple ML models
- Save the best model and preprocessor

### Running the Web Application

Start the FastAPI server:

```bash
python app.py
```

Or use FastAPI CLI:

```bash
fastapi dev app.py
```

Navigate to `http://localhost:8000` to access the prediction interface.

## 🤖 Machine Learning Models

The project evaluates the following regression models:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Support Vector Regressor (SVR)
5. K-Nearest Neighbors Regressor
6. Decision Tree Regressor
7. Random Forest Regressor
8. AdaBoost Regressor
9. Gradient Boosting Regressor
10. XGBoost Regressor
11. CatBoost Regressor

Each model undergoes hyperparameter tuning using GridSearchCV with 5-fold cross-validation, and the best performing model is selected based on R² score.

## 📊 Dataset Features

**Input Features:**
- `gender`: Student's gender
- `race_ethnicity`: Student's racial/ethnic group
- `parental_level_of_education`: Highest education level of parents
- `lunch`: Type of lunch (standard/free or reduced)
- `test_preparation_course`: Whether student completed test prep course
- `reading_score`: Score in reading test
- `writing_score`: Score in writing test

**Target Variable:**
- `math_score`: Score in math test (to be predicted)

## 🔄 ML Pipeline

### 1. Data Ingestion
- Reads raw data from CSV
- Splits into train (70%) and test (30%) sets
- Saves processed datasets

### 2. Data Transformation
- **Numerical Features**: Median imputation + Standard scaling
- **Categorical Features**: Most frequent imputation + One-Hot encoding + Scaling
- Saves preprocessing pipeline as pickle file

### 3. Model Training
- Trains multiple models with hyperparameter tuning
- Evaluates using R² score
- Saves the best performing model

### 4. Prediction
- Loads trained model and preprocessor
- Accepts new data via web interface
- Returns predicted math score

## 📈 Model Evaluation

Models are evaluated using:
- **R² Score**: Measures the proportion of variance explained by the model
- **Cross-Validation**: 5-fold CV during hyperparameter tuning

## 🛠️ Development

The project uses a modular architecture with:
- **Component-based design**: Separate modules for each ML pipeline stage
- **Custom logging**: Tracks execution and errors
- **Custom exceptions**: Better error handling and debugging
- **Configuration dataclasses**: Clean configuration management

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Saikat**
- GitHub: [@saikatcodec](https://github.com/saikatcodec)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or feedback, please reach out through GitHub issues.

---

**Note**: Make sure to have your dataset available at `notebooks/data/stud.csv` before running the training pipeline.
