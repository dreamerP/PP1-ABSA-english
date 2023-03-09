from itertools import product
import inflection
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils_for_aspect_extraction import get_tag
from textblob import TextBlob

negativeWordSet = {"don't", "never", "nothing", "nowhere", "noone", "none", "not",
                   "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't", "n't"
                                                                                  "wouldn't", "don't", "doesn't",
                   "didn't", "isn't", "aren't", "ain't", "no", "'t", "wont", "does n't"}


def pos_neg_wrds():
    """
    Esta funcion se utiliza para leer las palabras positivas y negativas de las listas auxiliares
    :return: una lista con palabras positivas y otra con apalabras negativas
    """
    pos_sentiments = []
    neg_sentiments = []
    pos_f = open("positive-words.txt", "r")
    for p in pos_f:
        pos_sentiments.append(p)
    neg_f = open("negative-words.txt", "r")
    for n in neg_f:
        neg_sentiments.append(n)
    return neg_sentiments, pos_sentiments


neg_sentiments, pos_sentiments = pos_neg_wrds()


class Aspect_Reviews_classes:
    def __init__(self):
        self.Pos_reviews = []
        self.Neg_reviews = []


def get_orientation(opinion_word):
    """
    Esta funcion se utiliza para obtener la polaridad de una palabra utilizado SentiWordNet
    :param opinion_word: palabra de la que se desea conocer la polaridad
    :return: polaridad
    """
    if opinion_word[:2] == "un":
        lst = list(swn.senti_synsets(opinion_word[2:len(opinion_word)]))
        neg = []
        pos = []
        if len(lst) != 0:
            for s in lst:
                neg.append(s.neg_score())
                pos.append(s.pos_score())
            if max(neg) > max(pos):
                return '+'
            elif max(neg) < max(pos):
                return '-'
            elif max(neg) == max(pos):
                return '='
    else:
        lst = list(swn.senti_synsets(opinion_word))
        neg = []
        pos = []
        if len(lst) != 0:
            for s in lst:
                neg.append(s.neg_score())
                pos.append(s.pos_score())
            if max(neg) > max(pos):
                return '-'
            elif max(neg) < max(pos):
                return '+'
            else:
                return '='


# this function is to calculate the semantic similarity score between two noun words input: word 1 & word2 ,
# output: wu-palmer semantic similarity score
def get_ratio212(x, y):
    sem1, sem2 = wordnet.synsets(x, pos='n'), wordnet.synsets(y, pos='n')
    maxscore = 0
    for i, j in list(product(*[sem1, sem2])):
        score = i.wup_similarity(j)
        maxscore = score if maxscore < score else maxscore
    return maxscore


# this function is to calculate the semantic similarity score between two adjectives input: word 1 & word2 ,
# output: wu-palmer semantic similarity score
def get_semantic_similarity_adejectives(x, y):
    sem1, sem2 = wordnet.synsets(x, pos='a'), wordnet.synsets(y, pos='a')
    maxscore = 0
    for i, j in list(product(*[sem1, sem2])):
        score = i.wup_similarity(j)
        if score is None:
            continue
        else:
            maxscore = score if maxscore < score else maxscore
    return maxscore


def check_for_trigram(b, c, aspect_opnions):
    posible_tags_for_opinion_words = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBG', 'VB', 'VBP', 'VBZ',
                                      'VBN', 'VBD', 'IN']
    cheked_for_append = []

    tag = get_tag(b)
    if tag in posible_tags_for_opinion_words:
        if (get_tag(c) != 'NN') or (get_tag(c) != 'NNS') or (get_tag(c) != 'NP'):

            if b not in aspect_opnions:
                cheked_for_append.append(b)
            if get_tag(c) in posible_tags_for_opinion_words:

                if c not in aspect_opnions:
                    cheked_for_append.append(c)
        elif get_tag(c) == 'NN' or get_tag(c) == 'NNS':
            '''orr = get_orientation(c)
            if orr != '=':'''
            if c not in aspect_opnions:
                cheked_for_append.append(c)

    return cheked_for_append


def check_for_bigram(j, aspect_opnions):
    posible_tags_for_opinion_words = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBG', 'VB', 'VBP', 'VBZ',
                                      'VBN', 'VBD', 'IN']
    tag = get_tag(j)
    if tag in posible_tags_for_opinion_words:
        if j not in aspect_opnions:
            return True
    elif (tag == 'NN') or (tag == 'NNS'):

        if j not in aspect_opnions:
            return True
    return False


def check_if_word_negative_word_is_before_positive(sentence, negative_word, aspect_opnions):
    if sentence.index(negative_word) < sentence.index(aspect_opnions[0]):
        if negative_word == "not":
            if ("only" in sentence) and (((sentence.index("only")) - (sentence.index("not"))) == 1):
                return True
    else:
        return False


def counter_negatives_and_positives_opinion_words(aspect_opinions, sentence):
    counter_positive = 0
    counter_negative = 0
    found_neg = False
    for opinion in aspect_opinions:
        polarity = get_polarity_using_vader(opinion)
        positive_word_from_list_found = False
        negative_word_from_list_found = False
        exist = False
        if polarity == '+' or polarity == '=':
            for positive_word in pos_sentiments:
                if positive_word.strip() in sentence:
                    positive_word_from_list_found = True
                    break
        elif polarity == '-' or polarity == '=':
            for negative_word in neg_sentiments:
                if negative_word.strip() in sentence:
                    negative_word_from_list_found = True
                    break
            for word in sentence:
                for negative_word in negativeWordSet:
                    if word == negative_word:
                        found_neg = True
                        exist = True
                        break
                    else:
                        found_neg = False
                if exist:
                    break
        if ((polarity == '+') and (positive_word_from_list_found == True)) or (
                (polarity != '+') and (positive_word_from_list_found == True)):
            if not found_neg:
                counter_positive += 1
            elif found_neg:
                if opinion in sentence:
                    if sentence.index(negative_word) < sentence.index(opinion):
                        if negative_word == "not":
                            if ("only" in sentence) and (((sentence.index("only")) - (
                                    sentence.index("not"))) == 1):
                                counter_positive += 1
                            else:
                                counter_negative += 1
                        else:
                            counter_negative += 1
                    elif sentence.index(negative_word) > sentence.index(opinion):
                        counter_positive += 1
        elif ((polarity == '-') and (negative_word_from_list_found == True)) or (
                (polarity != '-') and (negative_word_from_list_found == True)):

            if not found_neg:
                counter_negative += 1
            elif found_neg:
                if opinion in sentence:
                    if sentence.index(negative_word) < sentence.index(opinion):
                        if negative_word == "not":
                            if ("only" in sentence) and (((sentence.index("only")) - (
                                    sentence.index("not"))) == 1):
                                counter_negative += 1
                            else:
                                counter_positive += 1
                        else:
                            counter_positive += 1
                    elif sentence.index(negative_word) > sentence.index(opinion):
                        counter_negative += 1
    return counter_negative, counter_positive


def when_there_is_one_aspect_opinion(sentence, aspect_opnions, review):
    pos_lst = []
    neg_lst = []
    polarity = get_polarity_using_vader(aspect_opnions[0])
    postv = False
    negtv = False
    exist = False
    if polarity == '+' or polarity == '=':
        for p in pos_sentiments:
            if p.strip() in sentence:
                postv = True
                break
    elif polarity == '-' or polarity == '=':
        for n in neg_sentiments:
            if n.strip() in sentence:
                negtv = True
                break
    for a in sentence:
        for b in negativeWordSet:
            if a == b:
                found_neg = True
                neg_word = b
                exist = True
                break
            else:
                found_neg = False
        if exist:
            break
    if ((polarity == '+') and (postv == True)) or ((polarity != '+') and (postv == True)):

        if not found_neg:
            pos_lst.append(review)

        elif found_neg and aspect_opnions[0] in sentence:
            if sentence.index(neg_word) < sentence.index(sentence[0]):
                if neg_word == "not":
                    if ("only" in sentence) and (
                            ((sentence.index("only")) - (
                                    sentence.index("not"))) == 1):
                        pos_lst.append(review)


                    else:

                        neg_lst.append(review)

                else:

                    neg_lst.append(review)

            elif sentence.index(neg_word) > sentence.index(aspect_opnions[0]):

                pos_lst.append(review)

    elif ((polarity == '-') and (negtv == True)) or ((polarity != '-') and (negtv == True)):

        if not found_neg and aspect_opnions[0] in sentence:

            neg_lst.append(review)

        elif found_neg:
            if sentence.index(neg_word) < sentence.index(aspect_opnions[0]):
                if neg_word == "not":
                    if ("only" in sentence) and (
                            ((sentence.index("only")) - (
                                    sentence.index("not"))) == 1):

                        neg_lst.append(review)

                    else:

                        pos_lst.append(review)

                else:

                    pos_lst.append(review)

            elif sentence.index(neg_word) > sentence.index(aspect_opnions[0]):

                neg_lst.append(review)

    elif (polarity == '=') and (postv == True):

        pos_lst.append(review)

    elif (polarity == '=') and (negtv == True):
        neg_lst.append(review)

    return neg_lst, pos_lst


def when_there_is_none_aspect_opinion(sentence, aspect_opnions, review):
    neg_lst = []
    exist = False
    for a in sentence:
        for b in negativeWordSet:
            if a == b:
                found_neg = True
                neg_word = b
                exist = True
                break
            else:
                found_neg = False
            if exist:
                exist = True
    for op in aspect_opnions:
        if found_neg:
            if sentence.index(neg_word) < sentence.index(op):
                neg_lst.append(review)

    return neg_lst


def if_aspect_opinions_is_empty(sent, aspect_synonym_list, aspect_synonym, review):
    exist = False
    negative_review = []
    found_neg_p = False
    for i in sent:
        for j in negativeWordSet:
            if i == j:
                found_neg_p = True
                neg_word = j
                exist = True
                break
            else:
                found_neg_p = False
        if exist:
            break
    if found_neg_p:
        if len(aspect_synonym_list) == 1:
            if aspect_synonym in sent:
                if sent.index(neg_word) < sent.index(aspect_synonym):
                    negative_review.append(review)

            if inflection.pluralize(aspect_synonym) in sent:
                if sent.index(neg_word) < sent.index(inflection.pluralize(aspect_synonym)):
                    negative_review.append(review)

        elif len(aspect_synonym_list) > 1:

            if aspect_synonym in sent:

                if sent.index(neg_word) < sent.index(aspect_synonym_list[0]):
                    negative_review.append(review)

            if inflection.pluralize(aspect_synonym) in sent:
                if (sent.index(neg_word) < sent.index(
                        inflection.pluralize(aspect_synonym_list[0]))):
                    negative_review.append(review)
    return negative_review


def get_polarity_using_vader(word):
    sentiment = SentimentIntensityAnalyzer()
    polarity_scores = sentiment.polarity_scores(word.strip())
    if polarity_scores['pos'] > polarity_scores['neg'] > polarity_scores['neu']:
        return '+'
    if polarity_scores['neg'] > polarity_scores['pos'] > polarity_scores['neu']:
        return '-'
    else:
        return '='


def get_polarity_using_text_blob(word):
    polarity_score = TextBlob(word.strip()).sentiment.polarity
    if polarity_score > 0:
        return '+'
    elif polarity_score < 0:
        return '-'
    else:
        return '='
