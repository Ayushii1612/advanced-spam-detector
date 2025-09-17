import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc, classification_report, 
                           precision_recall_curve, average_precision_score, f1_score,
                           accuracy_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV

# Advanced ML libraries (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

# NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic text processing.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Spacy
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        SPACY_AVAILABLE = True
    except:
        SPACY_AVAILABLE = False
        nlp = None
        print("Spacy not available. Using basic tokenization.")

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced feature extraction for text data"""
    
    def __init__(self):
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.sia = SentimentIntensityAnalyzer()
        else:
            # Basic English stop words
            self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                              'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                              'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                              'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                              'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                              'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                              'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                              'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                              'under', 'again', 'further', 'then', 'once'}
            self.sia = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            feature_dict = self._extract_features(text)
            features.append(list(feature_dict.values()))
        return np.array(features)
    
    def _extract_features(self, text):
        """Extract comprehensive linguistic features"""
        features = {}
        
        if not isinstance(text, str):
            text = str(text)
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = max(1, len(re.split(r'[.!?]+', text)))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation and special characters
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(1, len(text))
        features['special_char_ratio'] = sum(1 for c in text if c in string.punctuation) / max(1, len(text))
        
        # URL and contact features
        features['url_count'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features['email_count'] = len(re.findall(r'\S+@\S+', text))
        features['phone_count'] = len(re.findall(r'\b\d{10,}\b', text))
        
        # Money and numbers
        features['money_mention'] = len(re.findall(r'\$\d+|\d+\s*(?:dollar|pound|euro|money|cash|prize|win|free)', text.lower()))
        features['number_count'] = len(re.findall(r'\b\d+\b', text))
        
        # Urgency and spam keywords
        urgency_words = ['urgent', 'immediate', 'asap', 'hurry', 'limited', 'expires', 'deadline', 'act now']
        spam_words = ['free', 'win', 'winner', 'congratulations', 'prize', 'offer', 'deal', 'discount', 'sale']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text.lower())
        features['spam_score'] = sum(1 for word in spam_words if word in text.lower())
        
        # Sentiment analysis
        if NLTK_AVAILABLE and self.sia:
            try:
                sentiment = self.sia.polarity_scores(text)
                features['sentiment_positive'] = sentiment['pos']
                features['sentiment_negative'] = sentiment['neg']
                features['sentiment_neutral'] = sentiment['neu']
                features['sentiment_compound'] = sentiment['compound']
            except:
                features.update({'sentiment_positive': 0, 'sentiment_negative': 0, 
                               'sentiment_neutral': 0, 'sentiment_compound': 0})
        elif TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                features['sentiment_positive'] = max(0, blob.sentiment.polarity)
                features['sentiment_negative'] = max(0, -blob.sentiment.polarity)
                features['sentiment_neutral'] = 1 - abs(blob.sentiment.polarity)
                features['sentiment_compound'] = blob.sentiment.polarity
            except:
                features.update({'sentiment_positive': 0, 'sentiment_negative': 0, 
                               'sentiment_neutral': 0, 'sentiment_compound': 0})
        else:
            features.update({'sentiment_positive': 0, 'sentiment_negative': 0, 
                           'sentiment_neutral': 0, 'sentiment_compound': 0})
        
        # Readability (simplified)
        words = text.split()
        if words:
            avg_sentence_length = len(words) / max(1, features['sentence_count'])
            features['readability_score'] = min(avg_sentence_length, 50)  # Cap at 50
        else:
            features['readability_score'] = 0
            
        return features

class AdvancedSpamDetector:
    """Advanced spam detection system with ensemble methods and optional deep learning"""
    
    def __init__(self, use_deep_learning=False):
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        self.models = {}
        self.ensemble_model = None
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.feature_extractor = AdvancedFeatureExtractor()
        self.feature_scaler = MinMaxScaler()  # Use MinMaxScaler for non-negative scaling
        self.tokenizer = None
        self.max_sequence_length = 100
        self.vocabulary_size = 10000
        self.deep_model = None
        
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs but keep placeholder
        text = re.sub(r'http\S+|www\S+|https\S+', ' url_placeholder ', text, flags=re.MULTILINE)
        
        # Remove emails but keep placeholder
        text = re.sub(r'\S+@\S+', ' email_placeholder ', text)
        
        # Remove phone numbers but keep placeholder
        text = re.sub(r'\b\d{10,}\b', ' phone_placeholder ', text)
        
        # Normalize repeated characters (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        # Advanced lemmatization with spacy if available
        if SPACY_AVAILABLE and nlp:
            try:
                doc = nlp(text)
                tokens = []
                for token in doc:
                    if not token.is_stop and not token.is_punct and token.lemma_.isalpha() and len(token.lemma_) > 2:
                        tokens.append(token.lemma_)
                text = " ".join(tokens)
            except:
                # Fallback to basic processing
                text = self._basic_text_cleaning(text)
        else:
            # Fallback: simple cleaning
            text = self._basic_text_cleaning(text)
        
        return text.strip()
    
    def _basic_text_cleaning(self, text):
        """Basic text cleaning when advanced NLP libraries are not available"""
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove stop words
        if NLTK_AVAILABLE:
            stop_words = set(stopwords.words('english'))
        else:
            stop_words = self.feature_extractor.stop_words
            
        tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    
    def create_deep_learning_model(self):
        """Create optimized neural network for text classification"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Simpler, more stable architecture
        model = Sequential([
            Embedding(self.vocabulary_size, 64, input_length=self.max_sequence_length),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Use a lower learning rate for more stable training
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, texts, labels):
        """Train the advanced spam detection system"""
        print("Starting advanced spam detection training...")
        
        # Convert to numpy arrays
        texts = np.array(texts)
        labels = np.array(labels)
        
        # Preprocess texts
        print("Preprocessing texts...")
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Feature extraction
        print("Extracting features...")
        
        # TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_tfidf_train = self.tfidf_vectorizer.fit_transform(X_train).toarray()
        X_tfidf_val = self.tfidf_vectorizer.transform(X_val).toarray()
        
        # Count features (character n-grams)
        self.count_vectorizer = CountVectorizer(
            max_features=2000,
            ngram_range=(2, 4),
            analyzer='char',
            min_df=2
        )
        X_count_train = self.count_vectorizer.fit_transform(X_train).toarray()
        X_count_val = self.count_vectorizer.transform(X_val).toarray()
        
        # Advanced linguistic features
        print("Extracting linguistic features...")
        X_features_train = self.feature_extractor.fit_transform(X_train)
        X_features_val = self.feature_extractor.transform(X_val)
        
        # Scale features to [0,1] range for compatibility with naive bayes
        X_features_train = self.feature_scaler.fit_transform(X_features_train)
        X_features_val = self.feature_scaler.transform(X_features_val)
        
        # Combine all features
        print("Combining features...")
        X_combined_train = np.hstack([X_tfidf_train, X_count_train, X_features_train])
        X_combined_val = np.hstack([X_tfidf_val, X_count_val, X_features_val])
        
        # Handle class imbalance
        if IMBALANCED_LEARN_AVAILABLE:
            print("Balancing dataset with SMOTE...")
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X_combined_train, y_train)
        else:
            print("SMOTE not available, using original dataset...")
            X_balanced, y_balanced = X_combined_train, y_train
        
        # Train individual models
        print("Training individual models...")
        
        # Multinomial Naive Bayes (works with non-negative features)
        print("Training MultinomialNB...")
        self.models['MultinomialNB'] = MultinomialNB(alpha=0.1)
        self.models['MultinomialNB'].fit(X_balanced, y_balanced)
        
        # Logistic Regression
        print("Training LogisticRegression...")
        self.models['LogisticRegression'] = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.models['LogisticRegression'].fit(X_balanced, y_balanced)
        
        # Random Forest
        print("Training RandomForest...")
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.models['RandomForest'].fit(X_balanced, y_balanced)
        
        # Calibrated SVM
        print("Training CalibratedSVM...")
        svm_base = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
        self.models['CalibratedSVM'] = CalibratedClassifierCV(svm_base, cv=3)
        self.models['CalibratedSVM'].fit(X_balanced, y_balanced)
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            self.models['XGBoost'].fit(X_balanced, y_balanced)
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            )
            self.models['LightGBM'].fit(X_balanced, y_balanced)
        
        # Create ensemble model
        print("Creating ensemble model...")
        estimators = [(name, model) for name, model in self.models.items()]
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        self.ensemble_model.fit(X_balanced, y_balanced)
        
        # Train deep learning model (if available)
        if self.use_deep_learning:
            print("Training deep learning model...")
            self.tokenizer = Tokenizer(num_words=self.vocabulary_size, oov_token='<UNK>')
            self.tokenizer.fit_on_texts(X_train)
            
            X_seq_train = self.tokenizer.texts_to_sequences(X_train)
            X_seq_val = self.tokenizer.texts_to_sequences(X_val)
            
            X_seq_train = pad_sequences(X_seq_train, maxlen=self.max_sequence_length)
            X_seq_val = pad_sequences(X_seq_val, maxlen=self.max_sequence_length)
            
            self.deep_model = self.create_deep_learning_model()
            
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(patience=3, factor=0.5)
            ]
            
            self.deep_model.fit(
                X_seq_train, y_train,
                validation_data=(X_seq_val, y_val),
                epochs=20,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
        
        # Evaluate models
        self._evaluate_models(X_combined_val, y_val, X_val)
        
        print("Training completed!")
    
    def _evaluate_models(self, X_val, y_val, X_val_text):
        """Evaluate all trained models"""
        print("\n=== Model Evaluation ===")
        
        results = {}
        
        # Evaluate traditional models
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                results[name] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred),
                    'recall': recall_score(y_val, y_pred),
                    'f1': f1_score(y_val, y_pred),
                    'auc': auc(*roc_curve(y_val, y_prob)[:2]) if len(np.unique(y_val)) > 1 else 0.5
                }
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        # Evaluate ensemble
        try:
            y_pred_ensemble = self.ensemble_model.predict(X_val)
            y_prob_ensemble = self.ensemble_model.predict_proba(X_val)[:, 1]
            
            results['Ensemble'] = {
                'accuracy': accuracy_score(y_val, y_pred_ensemble),
                'precision': precision_score(y_val, y_pred_ensemble),
                'recall': recall_score(y_val, y_pred_ensemble),
                'f1': f1_score(y_val, y_pred_ensemble),
                'auc': auc(*roc_curve(y_val, y_prob_ensemble)[:2]) if len(np.unique(y_val)) > 1 else 0.5
            }
        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
        
        # Evaluate deep learning model
        if self.use_deep_learning and self.deep_model and self.tokenizer:
            try:
                X_seq_val = self.tokenizer.texts_to_sequences(X_val_text)
                X_seq_val = pad_sequences(X_seq_val, maxlen=self.max_sequence_length)
                
                y_prob_deep = self.deep_model.predict(X_seq_val, verbose=0).flatten()
                y_pred_deep = (y_prob_deep > 0.5).astype(int)
                
                results['Deep Learning'] = {
                    'accuracy': accuracy_score(y_val, y_pred_deep),
                    'precision': precision_score(y_val, y_pred_deep),
                    'recall': recall_score(y_val, y_pred_deep),
                    'f1': f1_score(y_val, y_pred_deep),
                    'auc': auc(*roc_curve(y_val, y_prob_deep)[:2]) if len(np.unique(y_val)) > 1 else 0.5
                }
            except Exception as e:
                print(f"Error evaluating deep learning model: {e}")
        
        # Display results
        if results:
            results_df = pd.DataFrame(results).T
            print(results_df.round(4))
            
            # Simple plot
            try:
                plt.figure(figsize=(12, 8))
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
                x = np.arange(len(results_df.index))
                width = 0.15
                
                for i, metric in enumerate(metrics):
                    if metric in results_df.columns:
                        plt.bar(x + i * width, results_df[metric], width, label=metric, alpha=0.8)
                
                plt.xlabel('Models')
                plt.ylabel('Score')
                plt.title('Model Performance Comparison')
                plt.xticks(x + width * 2, results_df.index, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting results: {e}")
        else:
            print("No models evaluated successfully.")
        
        return results
    
    def predict(self, text, model_type='ensemble', return_probability=True):
        """Make predictions with specified model"""
        if not isinstance(text, str) or not text.strip():
            return ('ham', 0.0) if return_probability else 'ham'
            
        processed_text = self.preprocess_text(text)
        
        try:
            if model_type.lower() == 'deep' and self.use_deep_learning and self.deep_model:
                # Deep learning prediction
                if not self.tokenizer:
                    raise ValueError("Deep learning model not properly trained")
                    
                seq = self.tokenizer.texts_to_sequences([processed_text])
                seq = pad_sequences(seq, maxlen=self.max_sequence_length)
                prob = float(self.deep_model.predict(seq, verbose=0)[0][0])
                pred = int(prob > 0.5)
                
            else:
                # Traditional ML prediction
                # Extract features
                X_tfidf = self.tfidf_vectorizer.transform([processed_text]).toarray()
                X_count = self.count_vectorizer.transform([processed_text]).toarray()
                X_features = self.feature_extractor.transform([processed_text])
                X_features = self.feature_scaler.transform(X_features)
                
                X_combined = np.hstack([X_tfidf, X_count, X_features])
                
                if model_type.lower() == 'ensemble' and self.ensemble_model:
                    pred = self.ensemble_model.predict(X_combined)[0]
                    prob = float(self.ensemble_model.predict_proba(X_combined)[0][1])
                elif model_type in self.models:
                    model = self.models[model_type]
                    pred = model.predict(X_combined)[0]
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_combined)[0]
                        prob = float(proba[1])
                    else:
                        prob = 0.7 if pred == 1 else 0.3
                else:
                    # Default to first available model
                    if self.models:
                        model_name, model = next(iter(self.models.items()))
                        pred = model.predict(X_combined)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_combined)[0]
                            prob = float(proba[1])
                        else:
                            prob = 0.7 if pred == 1 else 0.3
                    else:
                        pred, prob = 0, 0.5
            
            label = 'spam' if pred == 1 else 'ham'
            
            if return_probability:
                return label, prob
            return label
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return ('ham', 0.0) if return_probability else 'ham'
    
    def explain_prediction(self, text):
        """Provide explanation for the prediction"""
        try:
            features = self.feature_extractor._extract_features(text)
            label, confidence = self.predict(text)
            
            explanation = {
                'prediction': label,
                'confidence': confidence,
                'text_statistics': {
                    'character_count': features.get('char_count', 0),
                    'word_count': features.get('word_count', 0),
                    'sentence_count': features.get('sentence_count', 1)
                },
                'spam_indicators': {
                    'urgency_score': features.get('urgency_score', 0),
                    'spam_score': features.get('spam_score', 0),
                    'money_mentions': features.get('money_mention', 0),
                    'capital_ratio': features.get('capital_ratio', 0),
                    'exclamation_count': features.get('exclamation_count', 0)
                },
                'sentiment': {
                    'compound': features.get('sentiment_compound', 0),
                    'positive': features.get('sentiment_positive', 0),
                    'negative': features.get('sentiment_negative', 0)
                }
            }
            
            return explanation
        except Exception as e:
            print(f"Explanation error: {e}")
            return {
                'prediction': 'ham',
                'confidence': 0.5,
                'error': str(e)
            }

def load_data(url=None):
    """Load spam dataset"""
    if url is None:
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    
    try:
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df['text'].tolist(), df['label'].tolist()
    except Exception as e:
        print(f"Error loading data from URL: {e}")
        print("Using sample data for demonstration...")
        
        # Return more comprehensive sample data for testing
        sample_texts = [
            "Hey, how are you doing today?",
            "FREE! Win a $1000 gift card! Click here NOW! Limited time offer!",
            "Meeting scheduled for 3 PM tomorrow in conference room A",
            "URGENT: Your account will be closed! Act immediately! Call now!",
            "Thanks for the help with the project, really appreciated",
            "Congratulations! You've won a free vacation! Click to claim your prize!",
            "Can you pick up some groceries on your way home?",
            "Your loan has been approved! Call now to claim your money!",
            "Don't forget about the team meeting tomorrow at 2 PM",
            "WINNER! You have been selected for a cash prize of $5000!",
            "The weather looks nice today, want to go for a walk?",
            "SALE! 50% off everything! Buy now before it's too late!",
            "Happy birthday! Hope you have a wonderful day",
            "Text STOP to 12345 to win free money instantly!",
            "Reminder: Your appointment is scheduled for Friday",
            "ALERT: Suspicious activity on your account. Verify immediately!",
            "How was your vacation? Did you have a good time?",
            "You owe $500 in unpaid bills. Pay now to avoid legal action!",
            "Great job on the presentation today, everyone was impressed",
            "FREE trial - no credit card required! Sign up now!"
        ]
        sample_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        return sample_texts, sample_labels

def print_available_libraries():
    """Print status of optional libraries"""
    print("=== Library Status ===")
    print(f"NLTK: {'✓' if NLTK_AVAILABLE else '✗'}")
    print(f"Spacy: {'✓' if SPACY_AVAILABLE else '✗'}")
    print(f"TextBlob: {'✓' if TEXTBLOB_AVAILABLE else '✗'}")
    print(f"XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗'}")
    print(f"LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗'}")
    print(f"Imbalanced-learn: {'✓' if IMBALANCED_LEARN_AVAILABLE else '✗'}")
    print(f"TensorFlow: {'✓' if TENSORFLOW_AVAILABLE else '✗'}")
    print()

def main():
    """Main function to run the advanced spam detector"""
    print("=== Advanced Spam Detection System ===\n")
    
    # Print library status
    print_available_libraries()
    
    # Load data
    print("Loading dataset...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples")
    print(f"Spam ratio: {np.mean(labels):.2%}\n")
    
    # Initialize detector
    use_deep_learning = TENSORFLOW_AVAILABLE and input("Use deep learning? (y/n): ").lower().startswith('y')
    detector = AdvancedSpamDetector(use_deep_learning=use_deep_learning)
    
    # Train the system
    try:
        detector.train(texts, labels)
    except Exception as e:
        print(f"Training error: {e}")
        return
    
    # Interactive testing
    print("\n" + "="*50)
    print("=== Interactive Spam Detection ===")
    print("Available models:")
    if detector.models:
        for model_name in detector.models.keys():
            print(f"  - {model_name}")
    if detector.ensemble_model:
        print("  - ensemble (recommended)")
    if detector.deep_model:
        print("  - deep")
    print("\nEnter text to classify (type 'quit' to exit):")
    print("You can specify model with: [model_name] text")
    print("Example: ensemble This is a test message")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYour text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            # Parse model specification
            parts = user_input.split(' ', 1)
            if len(parts) == 2 and parts[0].lower() in [m.lower() for m in detector.models.keys()] + ['ensemble', 'deep']:
                model_type = parts[0]
                text_to_classify = parts[1]
            else:
                model_type = 'ensemble'
                text_to_classify = user_input
            
            # Get prediction
            label, confidence = detector.predict(text_to_classify, model_type)
            
            print(f"\n--- Results ---")
            print(f"Model: {model_type}")
            print(f"Prediction: {label.upper()}")
            print(f"Confidence: {confidence:.3f}")
            
            # Show predictions from other available models
            print(f"\n--- Other Models ---")
            for model_name in detector.models.keys():
                if model_name.lower() != model_type.lower():
                    try:
                        pred, conf = detector.predict(text_to_classify, model_name)
                        print(f"{model_name}: {pred.upper()} ({conf:.3f})")
                    except:
                        print(f"{model_name}: Error")
            
            # Deep learning prediction if available and not already used
            if detector.deep_model and model_type.lower() != 'deep':
                try:
                    deep_pred, deep_conf = detector.predict(text_to_classify, 'deep')
                    print(f"Deep Learning: {deep_pred.upper()} ({deep_conf:.3f})")
                except Exception as e:
                    print(f"Deep Learning: Error - {e}")
            
            # Explanation
            try:
                explanation = detector.explain_prediction(text_to_classify)
                print(f"\n--- Explanation ---")
                spam_indicators = explanation['spam_indicators']
                text_stats = explanation['text_statistics']
                sentiment = explanation['sentiment']
                
                print(f"Text length: {text_stats['word_count']} words, {text_stats['character_count']} characters")
                print(f"Spam signals: {spam_indicators['spam_score']} spam words, {spam_indicators['urgency_score']} urgency words")
                if spam_indicators['money_mentions'] > 0:
                    print(f"Money mentions: {spam_indicators['money_mentions']}")
                if spam_indicators['exclamation_count'] > 0:
                    print(f"Exclamation marks: {spam_indicators['exclamation_count']}")
                if spam_indicators['capital_ratio'] > 0.3:
                    print(f"High capital letter ratio: {spam_indicators['capital_ratio']:.2f}")
                
                if abs(sentiment['compound']) > 0.1:
                    sentiment_desc = "positive" if sentiment['compound'] > 0 else "negative"
                    print(f"Sentiment: {sentiment_desc} ({sentiment['compound']:.2f})")
                
            except Exception as e:
                print(f"Explanation error: {e}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()