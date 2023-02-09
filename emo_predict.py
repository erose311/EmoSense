from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load model
loaded_model = tf.keras.models.load_model('emotions.h5')

# Load the tokenizer
with open('emotions_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences


def predict_emotion(sentence):
    emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    test_sequences = get_sequences(tokenizer, [sentence])
    predicted_emotion = np.argmax(loaded_model.predict(np.expand_dims(test_sequences[0], axis=0))[0])
    return emotions[predicted_emotion]

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_api():
    '''
    To render output (emotion) on HTML GUI
    :return: dictionary with predicted emotion as value
    '''
    if request.method == 'POST':
        sentence = request.form['sentence']
        print("Received sentence:", sentence)
        if len(sentence) == 0:
            return jsonify({'predicted_emotion': "Please enter a sentence to continue."})
        predicted_emotion = predict_emotion(sentence)
        print("Emotion: ", predicted_emotion)
        return jsonify({'predicted_emotion': predicted_emotion})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
