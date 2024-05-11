from flask import Flask, request, render_template
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

clf = load('model.joblib')
vectorizer = load('vectorizer.joblib')
def preprocess_text(text):
   
    text = text.lower()

 
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    
    words = word_tokenize(text)

  
    words = [word for word in words if word not in stop_words]

   
    words = [stemmer.stem(word) for word in words]
   
       
    text = ' '.join(words)

    return text


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    preprocessed_text = preprocess_text(text)
    X = vectorizer.transform([preprocessed_text])
    y_pred = clf.predict(X)
    if y_pred[0]== 1:
        result = 'real'
    else:
        result = 'fake'
    return render_template('result.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True)