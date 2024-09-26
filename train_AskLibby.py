import numpy as np
import json
import pickle
import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import clean_up_sentence, bow


# Read the intents file for training
with open('intents.json') as data:
    intents = json.load(data)

vocab = []
tags = []
xy = []

ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = clean_up_sentence(pattern)
        vocab.extend(w)

        xy.append((w, intent['tag']))
        tags.append(intent['tag'])

vocab = [w for w in vocab if w not in ignore_words]
vocab = sorted(list(set(vocab)))

tags = sorted(list(set(tags)))

print("Number of training samples: ", len(xy))
print("Number of QnA Categories: ", len(tags))
print("Vocab size: ", len(vocab))
print("List of unique stemmed words: \n", vocab)


X_train = []
Y_train = []

for sample in xy:

    bag = []
    Y = [0] * len(tags)

    X = sample[0]

    for w in vocab:
        bag.append(1) if w in X else bag.append(0)

    # Creates one hot vector for the current tag
    Y[tags.index(sample[1])] = 1

    X_train.append(np.array(bag))
    Y_train.append(np.array(Y))

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# defining the model architecture
model = tf.keras.Sequential()
Input = tf.keras.Input(shape=(len(X_train[0]),))
model.add(Input)
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(len(Y_train[0]), activation="softmax"))

model.summary()


# compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.F1Score()]
              )

model.fit(
    x=X_train,
    y=Y_train,
    # batch_size = 8,
    epochs=500,
    verbose='auto',
    validation_split=None,
    validation_data=None,
    shuffle=True,
    validation_freq=1
)


tf.keras.models.save_model(model, 'asklibby_core.keras')
pickle.dump({'vocab': vocab, 'tags': tags}, open("training_data", "wb"))



#### Inference Example:

print("\nInference Example:")
print("Que: What are your opening hours?")
p = bow("What are your opening hours?", vocab)
print("bow:", p)

probs = model.predict(np.array([p]))

ind = np.argmax(probs[0])
tag = tags[ind]
print("Question belongs to category:", tag)





