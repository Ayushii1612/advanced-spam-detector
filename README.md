📧 Advanced Spam Detection System

A production-ready spam detection system featuring ensemble machine learning models, deep learning neural networks, and comprehensive text analysis. Combines traditional ML algorithms (XGBoost, LightGBM, SVM, Random Forest) with LSTM networks and advanced feature engineering to achieve high-accuracy email/SMS spam classification with detailed prediction explanations.
---

🛠 Tech Stack Used
1. Python 3.13 → Core programming language for the project.
2. pandas & numpy → Data manipulation, cleaning, and numerical computations.
3. matplotlib & seaborn → Data visualization and exploratory analysis.
4. scikit-learn → Machine learning models, evaluation metrics, and preprocessing.
5. imbalanced-learn (SMOTE) → Balancing imbalanced datasets.
6. xgboost & lightgbm → High-performance gradient boosting algorithms.
7. nltk → Tokenization, stopword removal, and stemming.
8. spacy → Lemmatization and advanced NLP processing.
9. textblob → Sentiment analysis and text polarity scoring.
10. TensorFlow / Keras (optional) → Deep learning-based spam detection.
11. joblib / pickle → Model persistence (saving & loading trained models).
12. ipywidgets & IPython → Interactive components for real-time testing.
---

✨ Features
1. Data Preprocessing → Cleans and prepares raw text with tokenization, stopword removal, and lemmatization.
2. Feature Extraction → Uses TF-IDF, sentiment, punctuation, and text length features.
3. Linguistic Analysis → Captures spam indicators like money mentions, exclamation marks, and urgency words.
4. Class Balancing → Applies SMOTE to handle imbalanced datasets.
5. Multiple Models → Trains Naive Bayes, Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM.
6. Ensemble Learning → Combines multiple models for higher accuracy and robustness.
7. Deep Learning Support → Optionally integrates TensorFlow/Keras for neural network-based detection.
8. Interactive Mode → Classifies custom user input in real time.
9. Explainable Results → Shows why a message is spam (spam words, sentiment, length, money mentions).
10. High Performance → Achieves >98% accuracy with ensemble models.
---

🛠 Installation
1. Clone the repository: git clone https://github.com/your-username/advanced-spam-detection.git
                         cd advanced-spam-detection
2. Install dependencies: pip install -r requirements.txt
3. Download SpaCy English model: python -m spacy download en_core_web_sm
---

▶️ Usage
1. Run the script: python spam_detection.py
2. Choose model type: Select ML ensemble (default) or deep learning.
3. Wait for training: System preprocesses, extracts features, balances data, and trains models.
4. View evaluation: Accuracy, precision, recall, F1, and AUC scores are displayed.
5. Test interactively: Enter any text message to classify as SPAM or HAM.
6. Check confidence: Each prediction includes confidence probability.
7. Try specific models: Optionally run predictions using LogisticRegression, Naive Bayes, etc.
8. Exit anytime: Type quit to end the session.
---

📂 Project Structure

📦 advanced-spam-detection
 ┣ 📜 spam_detection.py   # Main program
 
 ┣ 📜 requirements.txt    # Dependencies
 
 ┣ 📜 README.md           # Documentation
 
 ┗ 📂 dataset             # Dataset (if included)
 ---

📌 Future Improvements
1. Deploy as a Flask / FastAPI web app
2. Build a browser-based demo with Streamlit
3. Add real-time email/SMS detection integration
---



