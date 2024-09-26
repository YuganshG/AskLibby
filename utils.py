import nltk
from nltk.stem.lancaster import LancasterStemmer


nltk.download('punkt_tab')
stemmer = LancasterStemmer()

def clean_up_sentence(sentence):
    """method to tokenize and stem the sentence"""

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words

def bow(sentence, vocab):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(vocab)

    for s in sentence_words:
        for i, w in enumerate(vocab):
            if w == s:
                bag[i] = 1

    return bag