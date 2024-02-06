import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from transformers import TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import pipeline
import pandas as pd
import numpy as np
import en_core_web_sm

from keras.layers import LSTM, Dense, Dropout, Input, Embedding, Bidirectional, Lambda
from keras import Model
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import spacy


def preProcessing(doc):
    nlp = en_core_web_sm.load()
    text_without_stop_words = [t.text for t in nlp(doc) if not t.is_stop]
    text_without_stop_words = " ".join(text_without_stop_words)
    lemmas = [t.lemma_ for t in nlp(text_without_stop_words)]
    return " ".join(lemmas)


def load_Question(paths):
    columns = ['question', 'class']
    dataset = pd.DataFrame(columns=columns)

    for index, path in enumerate(paths):
        tmp_dataset = pd.read_csv(path,
                                  sep='\n\n',
                                  header=None,
                                  engine='python')
        tmp_dataset.columns = ['question']
        tmp_dataset['class'] = index+1
        dataset = pd.concat(
            [dataset, tmp_dataset],
            ignore_index=True)
    return dataset


def load_Text(paths):
    columns = ['text', 'class']
    texts_dataset = pd.DataFrame(columns=columns)

    for index, path in enumerate(paths):
        texts = pd.read_csv(path, sep='\n\n', header=None,
                            engine='python')[0].str.cat(sep=" ")
        tmp_dataset = pd.DataFrame([texts], columns=['text'])
        tmp_dataset.columns = ['text']
        tmp_dataset['class'] = index+1
        texts_dataset = pd.concat(
            [texts_dataset, tmp_dataset], ignore_index=True)

    return texts_dataset


def load_GloveEmbedding(path):
    embeddings_index = {}
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


def make_EmbeddingMatrix(word2idx, word2vec, num_tokens, embedding_dim):
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    hits = 0
    misses = 0

    for word, i in word2idx.items():
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    return embedding_matrix, hits, misses

def l1_norm(x): 
    return 1 - K.abs(x[0] - x[1])

def LSTM_SiameseNet(MAX_LEN, num_tokens, embedding_dim, embedding_matrix, units_count=256):
    first_sent_in = Input(shape=(MAX_LEN, ))
    second_sent_in = Input(shape=(MAX_LEN, ))
    
    embedding_layer = Embedding(input_dim=num_tokens,
                                output_dim=embedding_dim,
                                input_length=MAX_LEN,
                                trainable=True,
                                mask_zero=True)

    embedding_layer.build((1,))
    embedding_layer.set_weights([embedding_matrix])

    first_sent_embedding = embedding_layer(first_sent_in)
    second_sent_embedding = embedding_layer(second_sent_in)

    lstm = Bidirectional(LSTM(units=units_count,
                          return_sequences=False))

    first_sent_encoded = lstm(first_sent_embedding)
    second_sent_encoded = lstm(second_sent_embedding)
    
    merged = Lambda(function=l1_norm,
                output_shape=lambda x: x[0],
                name='L1_distance')([first_sent_encoded, second_sent_encoded])
    
    predictions = Dense(1, activation='sigmoid',
                    name='classification_layer')(merged)

    model = Model([first_sent_in, second_sent_in], predictions)
    return model

def SiameseNet(MAX_LEN, LSTM_SiameseNet, dense_units=32, class_count=9, dropout=0.2):
    # Fully Connected DenseLayers
    dense1 = Dense(dense_units, activation='tanh', name='dense1')
    dense2 = Dense(dense_units, activation='tanh', name='dense2')
    classifier = Dense(class_count, activation='softmax', name='classifier_layer')

    # Embeddings
    embedding_layer = LSTM_SiameseNet.layers[2]
    embedding_layer.trainable = False

    # LSTM    
    lstm = LSTM_SiameseNet.layers[3]  # bidirectional lstm
    lstm.trainable = False
    
    # input must be tokenized
    # Combining Model parts Together
    siamese_net = tf.keras.models.Sequential()
    siamese_net.add(Input(shape=(MAX_LEN, )))
    siamese_net.add(embedding_layer)
    siamese_net.add(lstm)
    siamese_net.add(dense1)
    siamese_net.add(Dropout(dropout))
    siamese_net.add(dense2)
    siamese_net.add(Dropout(dropout))
    siamese_net.add(classifier)

    return siamese_net

def modelLoader():
    model_checkpoint = "bert-base-cased"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_checkpoint = 'models/QA_model/'
    qa_model = TFAutoModelForQuestionAnswering.from_pretrained(
        model_checkpoint)

    model_checkpoint = 'models/QA_model/'
    QA_pipeline = pipeline("question-answering",
                           model=model_checkpoint, tokenizer=qa_tokenizer)

    siamese_net = load_model('models/siamese_complete_model/model.h5')
    return QA_pipeline, siamese_net


def dataLoader(paths=None):
    dataset = pd.DataFrame(columns=['text', 'class'])
    if paths == None:
        paths = [r'data\1_3g4G modem.txt',
                 r'data\2_Change of Ownership.txt',
                 r'data\3_Economical Packages.txt',
                 r'data\4_FTTH Service.txt',
                 r'data\5_Participate in auction.txt',
                 r'data\6_Postpaid SIM.txt',
                 r'data\7_Prepaid SIM.txt',
                 r'data\8_TD-LTE modem.txt']

    for index, path in enumerate(paths):
        texts = pd.read_csv(path, sep='\n\n', header=None,
                            engine='python')[0].str.cat(sep=" ")
        tmp_dataset = pd.DataFrame([texts], columns=['text'])
        tmp_dataset.columns = ['text']
        tmp_dataset['class'] = index+1
        dataset = pd.concat([dataset, tmp_dataset], ignore_index=True)

    return dataset


def wordTokenizer(text=None):
    vocab_size = 1000
    oov_token = '<UNK>'

    questions_dataset = pd.DataFrame(columns=['question', 'class'])
    question_paths = [r'questions\1_3g4G modem.txt',
                      r'questions\2_Change of Ownership.txt',
                      r'questions\3_Economical Packages.txt',
                      r'questions\4_FTTH Service.txt',
                      r'questions\5_Participate in auction.txt',
                      r'questions\6_Postpaid SIM.txt',
                      r'questions\7_Prepaid SIM.txt',
                      r'questions\8_TD-LTE modem.txt']

    for index, path in enumerate(question_paths):
        tmp_dataset = pd.read_csv(
            path, sep='\n\n', header=None, engine='python')
        tmp_dataset.columns = ['question']
        tmp_dataset['question'] = tmp_dataset['question'].map(preProcessing)
        tmp_dataset['class'] = index+1
        questions_dataset = pd.concat(
            [questions_dataset, tmp_dataset], ignore_index=True)
        questions_dataset = pd.DataFrame(columns=['question', 'class'])

    tokens = " ".join(questions_dataset['question']).split()
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(tokens)
    return tokenizer


def QA(question, dataset=None):
    MAX_LEN = 40
    MAX_LEN = 40
    pad_type = 'post'
    trunc_type = 'post'
    tokenizer = wordTokenizer()
    return ('--------------------------------')
    if dataset is None:
        dataset = dataLoader()
    QA_pipeline, siamese_net = modelLoader()

    preprocessed_question = preProcessing(question)

    question_tokenized = tokenizer.fit_on_texts([preprocessed_question])
    question_padded = pad_sequences(question_tokenized, padding=pad_type,
                                    truncating=trunc_type, maxlen=MAX_LEN)

    question_class_probs = siamese_net(question_padded)
    question_class = np.argmax(question_class_probs)
    context = dataset['text'][dataset['class'] == question_class].values[0]

    answer = QA_pipeline(question=question, context=context)
    return answer
