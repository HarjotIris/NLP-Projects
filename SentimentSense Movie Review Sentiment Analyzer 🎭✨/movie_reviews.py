import pandas as pd

path = r"C:\Desktop\NLP Projects\IMDB Dataset of 50K Movie Reviews\IMDB Dataset.csv"

df = pd.read_csv(path)

# print(df.head(), df.info())

# print(df.isnull().sum())

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# Download necessary resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')


def preprocess_text(text):
    # convert to lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))

    # tokenize(split into words)
    words = word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # join back into a single sentence
    return " ".join(words)

# apply cleaning function to the dataset
df["cleaned_reviews"] = df["review"].apply(preprocess_text)

# displaying some cleaned data
# print(df[["review", "cleaned_reviews"]].head())


from sklearn.feature_extraction.text import TfidfVectorizer

# initialize the tf ID vectorizer
vectorizer = TfidfVectorizer(max_features=5000) # limited to 5k words

# transform the cleaned reviews
X = vectorizer.fit_transform(df["cleaned_reviews"])


# convert to a numpy array
X = X.toarray()
# print("Shape of X: ", X.shape)


# convert the sentiments to binary labels
df["sentiment"] = df["sentiment"].map(
    {"positive": 1,
     "negative": 0}
)

# assign a target variable
y = df["sentiment"].values
# print("First 5 sentiment labels: ", y[:5])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy score : {accuracy: .2f}")


# testing on custom reviews
def predict_sentiment(review):
    review = preprocess_text(review) # clean
    review_vectorized = vectorizer.transform([review]) # convert to TF-IDF
    prediction = model.predict(review_vectorized)
    return "Positive" if prediction==1 else "Negative" 

# Test on custom reviews
# print(predict_sentiment("I really loved this amazing movie!"))
# print(predict_sentiment("This was the worst film I have ever seen."))
# print(predict_sentiment("It was okay, not great but not terrible."))


# trying random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")


# adjusting the decision boundary
def predict_sentiment_threshold(review, threshold=0.6):
    review = preprocess_text(review)
    review_vectorized = vectorizer.transform([review])
    probability = model.predict_proba(review_vectorized)[0][1]
    return "Positive" if probability > threshold else "Negative"

# test with new threshold
print(predict_sentiment_threshold("It was okay, not great but not terrible.", threshold=0.6))


import joblib

# Save the trained model
joblib.dump(model, "sentiment_model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")
