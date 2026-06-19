from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import re
from nltk.corpus import stopwords
import nltk
import matplotlib
matplotlib.use('Agg')  # Prevent display errors in Flask
import matplotlib.pyplot as plt
import os

# -------------------- Initialize Flask App --------------------
app = Flask(__name__)
app.secret_key = "secret123"

# -------------------- Download Stopwords --------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------- Load Models and Vectorizer --------------------
nb_model = joblib.load('models/models/naive_bayes_model.pkl')
lr_model = joblib.load('models/models/logistic_regression_model.pkl')
rf_model = joblib.load('models/models/random_forest_model.pkl')
vectorizer = joblib.load('models/models/tfidf_vectorizer.pkl')


# -------------------- Text Cleaning Function --------------------
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)


# -------------------- ROUTES --------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# -------------------- FORM ROUTE --------------------
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        problem = request.form['problem']

        # Clean the text
        cleaned_problem = clean_text(problem)
        data = vectorizer.transform([cleaned_problem])

        # Individual Model Predictions
        prediction_nb = nb_model.predict(data)[0]
        prediction_lr = lr_model.predict(data)[0]
        prediction_rf = rf_model.predict(data)[0]

        result_nb = 'Spam' if prediction_nb == 1 else 'Not Spam'
        result_lr = 'Spam' if prediction_lr == 1 else 'Not Spam'
        result_rf = 'Spam' if prediction_rf == 1 else 'Not Spam'

        # -------------------- Majority Voting --------------------
        predictions = [prediction_nb, prediction_lr, prediction_rf]
        majority_vote = 1 if predictions.count(1) >= 2 else 0
        final_result = 'Spam' if majority_vote == 1 else 'Not Spam'

        # -------------------- Pie Chart Generation --------------------
        labels = ['Spam', 'Not Spam']
        sizes = [predictions.count(1), predictions.count(0)]
        colors = ['#FFD700', '#000000']  # yellow and black theme
        explode = (0.1, 0)  # explode the Spam part slightly

        chart_path = os.path.join('static', 'chart.png')
        plt.figure(figsize=(4, 4))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=explode, shadow=True, textprops={'color': "white"})
        plt.title("Model Prediction Distribution", color='yellow')
        plt.savefig(chart_path, transparent=True)
        plt.close()

        # -------------------- Final Results --------------------
        result = {
            'Naive Bayes': result_nb,
            'Logistic Regression': result_lr,
            'Random Forest': result_rf,
            'Overall Result (Majority Vote)': final_result,
            'chart_path': chart_path
        }

        flash('Your problem has been submitted successfully!', 'success')
        return render_template('result.html', result=result)

    return render_template('form.html')


# -------------------- API ROUTE --------------------
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned_message = clean_text(message)
    data = vectorizer.transform([cleaned_message])

    prediction_nb = nb_model.predict(data)[0]
    prediction_lr = lr_model.predict(data)[0]
    prediction_rf = rf_model.predict(data)[0]

    result_nb = 'Spam' if prediction_nb == 1 else 'Not Spam'
    result_lr = 'Spam' if prediction_lr == 1 else 'Not Spam'
    result_rf = 'Spam' if prediction_rf == 1 else 'Not Spam'

    predictions = [prediction_nb, prediction_lr, prediction_rf]
    majority_vote = 1 if predictions.count(1) >= 2 else 0
    final_result = 'Spam' if majority_vote == 1 else 'Not Spam'

    result = {
        'Naive Bayes': result_nb,
        'Logistic Regression': result_lr,
        'Random Forest': result_rf,
        'Overall Result (Majority Vote)': final_result
    }

    return render_template('result.html', result=result)


# -------------------- Run Flask App --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5065)
