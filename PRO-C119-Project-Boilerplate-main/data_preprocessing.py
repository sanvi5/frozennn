from flask import Flask, render_template, request, jsonify
import prediction
import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

app = Flask(__name__)

nltk.download('punkt')

words=[] 
classes = [] 
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

ps = PorterStemmer()

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        stemmed_word = ps.stem(word.lower())  
        if stemmed_word not in ignore_words:  
            stem_words.append(stemmed_word)  
    return stem_words

def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        for pattern in intent['patterns']:  
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            pattern_word_tags_list.append((pattern_words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words) 
    stem_words = list(set(stem_words))
    stem_words.sort()
    classes.sort()

    return stem_words, classes, pattern_word_tags_list

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0] 
        bag_of_words = []
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        for word in stem_words:
            bag_of_words.append(1) if word in stemmed_pattern_word else bag_of_words.append(0)

        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        labels_encoding = list([0]*len(classes))  
        tag = word_tags[1]   
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    with open('stem_words.pkl', 'wb') as f:
        pickle.dump(stem_words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    review = request.json.get('customer_review')
    if not review:
        response = {'status': 'error',
                    'message': 'Empty Review'}
    else:
        sentiment, path = prediction.predict(review)
        response = {'status': 'success',
                    'message': 'Got it',
                    'sentiment': sentiment,
                    'path': path}
    return jsonify(response)

@app.route('/save', methods=['POST'])
def save():
    date = request.json.get('date')
    product = request.json.get('product')
    review = request.json.get('review')
    sentiment = request.json.get('sentiment')

    data_entry = f"{date},{product},{review},{sentiment}\n"

    with open('reviews.log', 'a') as file:
        file.write(data_entry)

    return jsonify({'status': 'success',
                    'message': 'Data Logged'})

if __name__ == "__main__":
    app.run(debug=True)
