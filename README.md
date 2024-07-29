# 📈 Trading Signal Validity Prediction Model

This project implements a machine learning model to predict the validity of generated buy/sell signals. The model leverages BiLSTM (Bidirectional Long Short-Term Memory) and LSTM (Long Short-Term Memory) neural networks, followed by Logistic Regression for final prediction.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 📝 Overview

This repository contains the code and resources to train and evaluate a model that predicts the validity of trading signals. The model is designed to improve the accuracy of buy/sell decisions by analyzing historical trading data.

## 🌟 Features

- **📈 Advanced Time-Series Analysis**: Utilizes BiLSTM and LSTM networks for deep temporal understanding.
- **🧠 Logistic Regression**: Applies logistic regression for binary classification.
- **📊 Performance Metrics**: Includes detailed performance metrics and evaluation scripts.

## 🏗️ Architecture

1. **Data Preprocessing**: Cleans and prepares trading data for model training.
2. **BiLSTM and LSTM Models**: Captures complex temporal patterns in the data.
3. **Logistic Regression**: Makes the final prediction on the validity of trading signals.

## 🛠️ Installation

To set up the environment for training and evaluating the model, follow these steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-username/trading-signal-validity-prediction.git
    cd trading-signal-validity-prediction
    ```

2. **Create a virtual environment and activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1. **Prepare your dataset**
   - Ensure your dataset is in the required format and place it in the `data/` directory.

2. **Run the main.ipynb to create models first time**
3. **Run retrain.ipynb to retrain the models again and again on new time**

## 🏋️‍♂️ Model Training

The model training process involves several steps:

1. **Data Preprocessing**: Handles missing values, scales features, and splits the data into training and testing sets.
2. **Model Training**: Trains the BiLSTM and LSTM models on the training data.
3. **Logistic Regression**: Applies logistic regression on the features extracted from the LSTM models.

## 📊 Evaluation

The model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Detailed evaluation results are stored in the `results/` directory.

## 🤝 Contributing

We welcome contributions from the community! Here’s how you can help:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Push your changes to your fork.
5. Create a pull request detailing your changes.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📫 Contact

For any inquiries or feedback, please contact:
- **Email**: nisargpatel0466@gmail.com 
- **GitHub**: nisarg30(https://github.com/nisarg30)

---

Happy trading and predicting! 🚀
