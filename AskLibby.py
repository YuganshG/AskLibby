import numpy as np
import json
import pickle
import tensorflow as tf
from utils import clean_up_sentence, bow


# import our chat-bot intents file
with open('intents.json') as data:
    intents = json.load(data)

data = pickle.load(open("training_data", "rb"))
vocab = data['vocab']
tags = data['tags']

# importing the saved model
asklibby = tf.keras.models.load_model("asklibby_core.keras")


# chatting with AskLibby

bolds = "\033[1m"
bolde = "\033[0m"

highs = "\033[1;32m"
highe = "\033[0m"

bot_name = "AskLibby"

print(f"Welcome! I’m {highs}{bot_name}{highe}, your friendly library assistant.\n(type 'quit' to exit)\n")

ERROR_THRESHOLD = 0.75

while True:

    sentence = input("You: ")
    if sentence == "quit":
        break

    x = bow(sentence, vocab)

    probs = asklibby.predict(np.array([x]), verbose=0)

    ind = np.argmax(probs[0])
    tag = tags[ind]

    prob = probs[0][ind]

    if prob > ERROR_THRESHOLD:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{highs}{bot_name}{highe}: {np.random.choice(intent['responses'])}")

        if tag == "goodbye":
            break
    else:
        print(
            f"{highs}{bot_name}{highe}: I do not understand...\n Want to know location/availability of a specific book? You can get that info in the catalog here:: [Library Catalog](http://example.com/library-catalog).")



