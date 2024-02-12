import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# nltk.download('stopwords')


# Filter 1: remove KeyPhrases Longer than 3 Words
def filter1(preds):
    final_preds = []
    for index, pred in enumerate(preds):

        final_preds.append(pred.copy())
        for kp in pred:
            try:
                length = len(kp.split())
            except:
                print(kp)
            if length > 3:

                final_preds[index].remove(kp)
    return final_preds


# Filter 2: Mixing One Words
def return_one_words(pred):
    ones = []
    for kp in pred:
        if len(kp.split()) == 1:

            ones.append(kp)
    return ones


def mix_ones(ones):

    output = {}

    if len(ones) > 1:
        for word1 in ones:
            for word2 in ones:
                if word1 != word2:
                    output[word1 + " " + word2] = [word1, word2]

    else:

        output = {None: None}
    return output


def filter2(preds, text_ngrams, word2score, threshold=0.45):
    output_preds = []
    for index, pred in enumerate(preds):
        output_preds.append(pred.copy())
        ones = return_one_words(pred)
        mixed = mix_ones(ones)

        for key in mixed.keys():

            if key in text_ngrams[index]:

                word1 = mixed[key][0]
                word2 = mixed[key][1]
                try:
                    word1_score = word2score[word1]
                    word2_score = word2score[word2]
                    candidate_score = word2score[key]

                    mean_score = threshold * (word1_score + word2_score)
                    if candidate_score >= mean_score:

                        try:

                            output_preds[index].remove(word1)
                            output_preds[index].remove(word2)
                            pass

                        except:
                            pass

                        if key not in output_preds[index]:

                            output_preds[index].append(key)

                except:
                    pass

    return output_preds


# Filter 3: KeyPhrase Subsets Remove
def filter3(preds, word2score):

    output_preds = []
    for index, pred in enumerate(preds):
        output_preds.append(pred.copy())
        for small_kp in pred:
            small_words = small_kp.split()
            for big_kp in pred:
                big_words = big_kp.split()

                if len(small_words) < len(big_words):
                    vectorizer = CountVectorizer(
                        ngram_range=(1, len(big_words)))
                    vectorizer.fit([big_kp])
                    big_ngrams = vectorizer.get_feature_names_out()
                    if small_kp in big_ngrams:
                        try:
                            small_score = word2score[small_kp]
                            big_score = word2score[big_kp]
                            remove_small = False
                            remove_big = False

                            if len(small_words) == 2:
                                if small_score > 4*big_score:
                                    remove_big = True

                                elif small_score < 1.8*big_score:
                                    remove_small = True

                            else:
                                if small_score*0.6 <= big_score:
                                    remove_small = True

                                elif small_score > 4.2*big_score:
                                    remove_big = True

                            if remove_small:
                                output_preds[index].remove(small_kp)
                            if remove_big:
                                output_preds[index].remove(big_kp)

                        except:
                            pass

    return output_preds


# Filter 4: mixing two words
def return_two_words(pred):
    twos = []
    for kp in pred:
        if len(kp.split()) == 2:
            twos.append(kp)
    return twos


def mix_twos(twos):
    output = {}
    if len(twos) > 1:
        for word1 in twos:
            for word2 in twos:
                if word1 != word2:
                    if word1.split()[1] == word2.split()[0]:
                        output[word1 + " " + word2.split()[1]] = [word1, word2]

    else:
        output = {None: None}
    return output


def filter4(preds, word2score, threshold=0.45):
    output_preds = []
    for index, pred in enumerate(preds):
        output_preds.append(pred.copy())
        twos = return_two_words(pred)
        mixed = mix_twos(twos)
        for key, value in mixed.items():

            try:

                word1 = value[0]
                word2 = value[1]
                word1_score = word2score[word1]
                word2_score = word2score[word2]
                mean_score = threshold * (word1_score + word2_score)
                candidate_score = word2score[key]
                if candidate_score >= mean_score:
                    try:
                        output_preds[index].remove(word1)
                        output_preds[index].remove(word2)
                        pass

                    except:
                        pass

                    if key not in output_preds[index]:

                        output_preds[index].append(key)

            except:
                pass
    return output_preds


def increment_score(kp, score):

    length = len(kp.split())
    factor1 = 0.045
    factor2 = 0.12
    factor3 = 0.03

    if length == 1:
        score += factor1
    if length == 2:
        score += factor2
    if length == 3:
        score += factor3
    return score


def candidate_score(features, tfidf_scores, candidates, page_count):
    chunk_indexes = []
    chunk_features = []
    chunk_scores = []
    feature_score = []
    for i in range(page_count):
        tmp_indexes = []
        tmp_features = []
        tmp_scores = []
        for index, feature in enumerate(features):
            if feature in candidates[i]:
                tmp_indexes.append(index)
                tmp_features.append(feature)
                score = tfidf_scores[i][index]
                score = increment_score(feature, score)
                tmp_scores.append(score)
        chunk_indexes.append(tmp_indexes)
        chunk_features.append(tmp_features)
        chunk_scores.append(tmp_scores)

    for i in range(len(chunk_features)):
        tmp = []
        for feature, score in zip(chunk_features[i], chunk_scores[i]):
            tmp.append((feature, score))
        feature_score.append(sorted(tmp, key=lambda x: x[1], reverse=True))

    return feature_score


def add_NE(name_entities, preds):
    for index, name_entity_lst in enumerate(name_entities):
        for name_entity in name_entity_lst:
            if name_entity not in preds[index]:
                preds[index].append(name_entity)
    return preds


def apply_filters(preds, descrete_ngrams, word2score):
    preds = filter1(preds)
    preds = filter2(preds, descrete_ngrams, word2score)
    preds = filter4(preds, word2score, threshold=0.4)
    # preds = filter3(preds, word2score)
    return preds


def predict_with_threshold(feature_score, descrete_ngrams, word2score, name_entities,
                           minimum=4, maximum=6, threshold=0.2):
    preds = []
    for pred in feature_score:
        tmp_pred = []
        for feature, score in pred:
            if score >= threshold:
                tmp_pred.append(feature)
        tmp_pred = apply_filters([tmp_pred], descrete_ngrams, word2score)[0]

        if len(tmp_pred) < minimum:
            tmp_pred = []
            for feature, _ in pred[:20]:
                tmp_pred.append(feature)
            tmp_pred = apply_filters([tmp_pred], descrete_ngrams, word2score)[
                0][:minimum]

        elif len(tmp_pred) > maximum:
            tmp_pred = []
            for feature, _ in pred[:40]:
                tmp_pred.append(feature)
            tmp_pred = apply_filters([tmp_pred], descrete_ngrams, word2score)[
                0][:maximum]

        preds.append(tmp_pred)
    preds = add_NE(name_entities, preds)
    # print(preds)
    preds = apply_filters(preds, descrete_ngrams, word2score)
    return preds


def increment_score(kp, score):
    length = len(kp.split())
    factor1 = 0.045
    factor2 = 0.12
    factor3 = 0.03

    if length == 1:
        score += factor1

    if length == 2:
        score += factor2

    if length == 3:
        score += factor3
    return score


def features_scores_tfidfmodel(corpus):
    tfidf = TfidfVectorizer(stop_words=stopwords.words(

        'english'), ngram_range=(1, 3))

    tfidf_scores = tfidf.fit_transform(corpus).toarray()

    features = tfidf.get_feature_names_out()
    return features, tfidf_scores, tfidf


def make_phrase2score(tfidf_scores, tfidf_model):
    phrase2score = {}
    scores = tfidf_scores.max(axis=0)
    for feature, index in tfidf_model.vocabulary_.items():

        score = scores[index]

        phrase2score[feature] = score
    return phrase2score


def make_candidates(corpus):
    tfidf = TfidfVectorizer(stop_words=stopwords.words(
        'english'), ngram_range=(1, 3))
    candidates = []
    for text in corpus:
        page_candidates = []
        try:
            tfidf.fit([text])
        except:
            candidates.append([])
            continue
        ngrams = tfidf.get_feature_names_out()
        for ngram in ngrams:
            if not ngram.replace(" ", "").isdigit():
                ngram = " ".join(
                    [word for word in ngram.split() if not word.isdigit()])
                page_candidates.append(ngram)
        candidates.append(page_candidates)
    return candidates


def extract_NE(corpus):
    nlp = spacy.load("en_core_web_sm")
    all_NE = []
    for text in corpus:
        NEs = nlp(text)
        all_NE.append(NEs)

    NE = []
    for NE_set in all_NE:
        tmp_entities = []
        for entity in NE_set.ents:
            entity_text = entity.text
            if entity.label_ in ['PERSON', 'LOCATION', 'ORG', 'FACILITY', 'PRODUCT', 'LANGUAGE', 'WORK_OF_ART', 'EVENT'] and len(entity_text) > 3 and len(entity_text.split()) < 5:
                # print(entity_text, entity.label_)
                tmp_entities.append(entity_text)
        NE.append(tmp_entities)
    return NE


def make_feature_score(features, tfidf_scores, candidates, pages):
    feature_score = candidate_score(
        features, tfidf_scores, candidates, page_count=pages)
    return feature_score


def extract(corpus, count=5):
    pages = len(corpus)
    candidates = make_candidates(corpus)
    features, tfidf_scores, tfidf_model = features_scores_tfidfmodel(corpus)
    phrase2score = make_phrase2score(tfidf_scores, tfidf_model)
    feature_score = make_feature_score(
        features, tfidf_scores, candidates, pages)
    NE = extract_NE(corpus)

    preds = predict_with_threshold(feature_score, candidates, phrase2score, NE,
                                   minimum=count, maximum=count, threshold=1)

    return preds
