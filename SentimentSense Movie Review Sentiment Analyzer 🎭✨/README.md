# 🎭 SentimentSense: Movie Review Sentiment Analyzer 🚀

## **Overview**
SentimentSense is a machine learning-based sentiment analysis model that predicts whether a given movie review is **positive** or **negative**. Using **Natural Language Processing (NLP)** and **Logistic Regression**, this model converts text into numerical form and learns sentiment patterns.

## **📂 Dataset**
- Source: IMDb movie reviews dataset
- Contains **50,000 reviews** labeled as **positive** or **negative**.

## **🛠️ Features**
✔ **Preprocessing**: Cleans reviews by removing punctuation, stopwords, and converting text to lowercase.  
✔ **TF-IDF Vectorization**: Converts text into numerical features for ML models.  
✔ **Model Training**: Uses **Logistic Regression** for classification.  
✔ **High Accuracy**: Achieved **~89% accuracy** on test data.  
✔ **Custom Review Testing**: Users can input a review to get predictions.

## **🚀 How It Works**
1. **Preprocesses Text** – Cleans and tokenizes reviews.
2. **Converts Text to Numbers** – Uses TF-IDF for feature extraction.
3. **Trains Model** – Learns sentiment patterns using Logistic Regression.
4. **Predicts Sentiment** – Classifies reviews as **positive or negative**.

## **📌 Installation & Usage**
### **1️⃣ Install Dependencies**
```bash
pip install pandas numpy scikit-learn nltk joblib
```

### **2️⃣ Run the Sentiment Analysis**
```python
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(review):
    cleaned_review = preprocess_text(review)  # Clean the text
    review_vectorized = vectorizer.transform([cleaned_review])  # Convert to TF-IDF
    prediction = model.predict(review_vectorized)[0]  # Get prediction (0 or 1)
    return "Positive" if prediction == 1 else "Negative"

# Example usage
print(predict_sentiment("This movie was fantastic!"))
```

## **🔍 Example Predictions**
| Review                                   | Prediction  |
|------------------------------------------|------------|
| "Absolutely loved this film!"             | **Positive** ✅ |
| "This was the worst movie ever."         | **Negative** ❌ |
| "It was okay, not great but not bad."    | **Negative** ❌ (Threshold tuning needed) |

## **💡 Future Improvements**
🔹 Fine-tune threshold to handle **neutral reviews** better.  
🔹 Experiment with **Random Forest or Deep Learning models** (LSTM).  
🔹 Deploy as a **FastAPI service** for real-world use.

---

✨ **Built with ❤️ by Harjot / Iris**  
📌 **Status:** ✅ Complete but open for improvements!
