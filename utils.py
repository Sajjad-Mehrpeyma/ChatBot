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

def preProcessing(doc):
    nlp = en_core_web_sm.load()
    text_without_stop_words = [t.text for t in nlp(doc) if not t.is_stop]
    text_without_stop_words = " ".join(text_without_stop_words)
    lemmas = [t.lemma_ for t in nlp(text_without_stop_words)]
    return " ".join(lemmas)

def modelLoader():
    model_checkpoint = "bert-base-cased"
    qa_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_checkpoint = 'models/QA_model/'
    qa_model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_checkpoint = 'models/QA_model/'
    QA_pipeline = pipeline("question-answering", model=model_checkpoint, tokenizer=qa_tokenizer)

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
        texts = pd.read_csv(path, sep='\n\n', header=None, engine='python')[0].str.cat(sep=" ")
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
        tmp_dataset = pd.read_csv(path, sep='\n\n', header=None, engine='python')
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
    if dataset is None:
        dataset = dataLoader()

    QA_pipeline, siamese_net = modelLoader()

    preprocessed_question = preProcessing(question)

    question_tokenized = tokenizer.fit_on_texts([preprocessed_question])
    question_padded = pad_sequences(question_tokenized, padding=pad_type,
                                    truncating=trunc_type, maxlen=MAX_LEN)

    question_class_probs = siamese_net(question_padded)
    question_class = np.argmax(question_class_probs)
    context = dataset['text'][dataset['class']==question_class].values[0]

    answer = QA_pipeline(question=question, context=context)
    return answer