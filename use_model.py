import pickle

from flask import Flask, render_template, request
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
model = load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequences = loaded_tokenizer.texts_to_sequences([text])
    max_sequence_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)[0][0]

    # Assuming your model is binary classification (0 or 1)
    is_cyberbullying = prediction >= 0.5
    result = "Cyberbullying Detected!" if is_cyberbullying else "No Cyberbullying Detected"
    result_color = "red" if is_cyberbullying else "green"

    return render_template('index.html', prediction_text=result, result_color=result_color)


if __name__ == '__main__':
    app.run(debug=True)
