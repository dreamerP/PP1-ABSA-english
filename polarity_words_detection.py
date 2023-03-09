import inflection
from nltk import ngrams
from utils_for_sentiment_clasification import Aspect_Reviews_classes, negativeWordSet, check_for_trigram, \
    check_for_bigram, \
    counter_negatives_and_positives_opinion_words, when_there_is_one_aspect_opinion, when_there_is_none_aspect_opinion
from utils_general import clean_the_review, remove_stop_words
from tqdm import tqdm

def classify_polarity_for_noun_aspects(reviews, table_noun_and_related_words):
    """
    Esta función toma los aspectos extraídos y clasifica las opiniones donde aparezcan esos aspectos de acuerdo
    a su polaridad
    :param reviews: opiniones sin etiquetar
    :param table_noun_and_related_words: aspectos extraídos de las opiniones
    :return: diccionario que contiene las reseñas por aspecto clasificadas
    """
    aspect_plus_it_reviews_table = {}
    for aspect in tqdm(table_noun_and_related_words, desc = 'Detectando polaridad para aspectos (sustantivos)'):
        positive_and_negative_reviews_for_aspect = Aspect_Reviews_classes()
        for review in reviews:
            neg_lst = []
            pos_lst = []
            if (review.find(aspect) != -1) or (review.find(inflection.pluralize(aspect)) != -1):
                sentences = review.split(",")
                for sent in sentences:
                    aspect_opnions = []  # Aqui se almacenan los términos de opinión que se utilizan para hacer
                                         # referencia a un aspecto
                    splitted_sentence = sent.split(" ")

                    if (aspect in splitted_sentence) or (inflection.pluralize(aspect) in splitted_sentence):
                        sentence = clean_the_review(sent).lower()
                        sentence = remove_stop_words(sentence)

                        if len(sentence) == 2:
                            bigrams = ngrams(sentence, 2)
                            for word1, word2 in bigrams:
                                if word1 == aspect or word1 == inflection.pluralize(aspect):
                                    if check_for_bigram(word2, aspect_opnions):
                                        aspect_opnions.append(word2)
                                elif word2 == aspect or word2 == inflection.pluralize(aspect):
                                    if check_for_bigram(word1, aspect_opnions):
                                        aspect_opnions.append(word1)

                        elif len(sentence) > 2:
                            Tgrams = ngrams(sentence, 3)
                            for a, b, c in Tgrams:
                                if a == aspect or a == inflection.pluralize(aspect):
                                    if len(check_for_trigram(b, c, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(b, c, aspect_opnions)

                                elif (b == aspect) or (b == inflection.pluralize(aspect)):
                                    if len(check_for_trigram(a, c, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(a, c, aspect_opnions)

                                elif c == aspect or c == inflection.pluralize(aspect):
                                    if len(check_for_trigram(a, b, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(a, b, aspect_opnions)

                                if (a == aspect) or (a == inflection.pluralize(aspect)):
                                    if len(check_for_trigram(c, b, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(c, b, aspect_opnions)

                                if (b == aspect) or (b == inflection.pluralize(aspect)):
                                    if len(check_for_trigram(c, a, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(c, a, aspect_opnions)

                                if (c == aspect) or (c == inflection.pluralize(aspect)):
                                    if len(check_for_trigram(b, a, aspect_opnions)) > 0:
                                        aspect_opnions = aspect_opnions + check_for_trigram(b, a, aspect_opnions)

                        if len(aspect_opnions) == 1:
                            neg, pos = when_there_is_one_aspect_opinion(sentence, aspect_opnions, review)
                            if len(neg) > 0:
                                neg_lst = neg_lst + neg
                            elif len(pos) > 0:
                                pos_lst = pos_lst + pos

                        elif len(aspect_opnions) > 1:
                            counter_negative, counter_positive = counter_negatives_and_positives_opinion_words(
                                aspect_opnions, sentence)
                            if counter_negative > counter_positive:
                                neg_lst.append(review)

                            elif counter_negative < counter_positive:
                                pos_lst.append(review)

                            elif counter_negative == counter_positive:
                                pos_lst.append(review)
                                neg_lst.append(review)

                        elif (len(aspect_opnions) == 0) and (sentence != "") and (aspect in sentence):
                            neg = when_there_is_none_aspect_opinion(sentence, aspect_opnions, review)
                            if len(neg) > 0:
                                neg_lst = neg_lst + neg

                if len(neg_lst) > len(pos_lst):
                    positive_and_negative_reviews_for_aspect.Neg_reviews.append(review)
                elif len(neg_lst) < len(pos_lst):
                    positive_and_negative_reviews_for_aspect.Pos_reviews.append(review)

        aspect_plus_it_reviews_table[aspect.rstrip()] = positive_and_negative_reviews_for_aspect
    return aspect_plus_it_reviews_table


def classify_polarity_for_noun_phrases(reviews, ph_aspects):
    """
    Esta función clasifica las opiniones donde se encuentren las noun phrases extraídas en positivas o negativas
    :param reviews: reseñas sin etiquetar
    :param ph_aspects: phrase aspects previamente extraídos
    :return: diccionario que contiene para cada noun phrase las reseñas positivas y negativas
    """
    found_neg = False
    aspect_table = {}
    for phrase_aspect in tqdm(ph_aspects, desc='Detectando polaridad para aspectos de tipo frases sustantivas'):
        positive_and_negative_reviews_for_aspect = Aspect_Reviews_classes()
        splitted_aspects = phrase_aspect.split()
        for review in reviews:
            neg_lst = []
            pos_lst = []
            rev = review
            if (rev.find(phrase_aspect) != -1) or (rev.find(inflection.pluralize(phrase_aspect)) != -1) or (
                    rev.find(splitted_aspects[0]) != -1) or (
                    rev.find(inflection.pluralize(splitted_aspects[0])) != -1) or (
                    (rev.find(splitted_aspects[1]) != -1) or (
                    rev.find(inflection.pluralize(splitted_aspects[1])) != -1)):
                sentences = rev.split(",")
                for sent in sentences:
                    aspect_opnions = []
                    splitted_sentence = sent.split()
                    sentnce = clean_the_review(sent).lower()
                    r_wrds = remove_stop_words(sentnce)
                    if (sent.find(phrase_aspect) != -1) or (sent.find(inflection.pluralize(phrase_aspect)) != -1):
                        thregrams = ngrams(r_wrds, 3)
                        for a, b, c in thregrams:
                            if ((a == splitted_aspects[0]) or (a == inflection.pluralize(splitted_aspects[0]))) and (
                                    (b == splitted_aspects[1]) or (b == inflection.pluralize(splitted_aspects[1]))):
                                if check_for_bigram(c, aspect_opnions):
                                    aspect_opnions.append(c)
                            elif ((b == splitted_aspects[0]) or (b == inflection.pluralize(splitted_aspects[0]))) and (
                                    (c == splitted_aspects[1]) or (c == inflection.pluralize(splitted_aspects[1]))):
                                if check_for_bigram(a, aspect_opnions):
                                    aspect_opnions.append(a)
                        if len(aspect_opnions) == 1:
                            neg, pos = when_there_is_one_aspect_opinion(sentnce, aspect_opnions, review)
                            if len(neg) > 0:
                                neg_lst = neg_lst + neg
                            elif len(pos) > 0:
                                pos_lst = pos_lst + pos
                        elif len(aspect_opnions) > 1:
                            counter_negatives, counter_positives = counter_negatives_and_positives_opinion_words(
                                aspect_opnions, splitted_sentence)
                            if counter_negatives > counter_positives:
                                neg_lst.append(rev)
                            elif counter_negatives < counter_positives:
                                pos_lst.append(rev)
                            elif counter_negatives == counter_positives:
                                neg_lst.append(rev)
                                pos_lst.append(rev)

                        elif (len(aspect_opnions) == 0) and (sentnce != "") and (sent.find(phrase_aspect) != -1):
                            exist = False
                            for a in splitted_sentence:
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
                            if found_neg:
                                if phrase_aspect[0] in splitted_sentence:
                                    if splitted_sentence.index(neg_word) < splitted_sentence.index(phrase_aspect[0]):
                                        neg_lst.append(rev)


                    elif (((splitted_aspects[0] in splitted_sentence) or (
                            inflection.pluralize(splitted_aspects[0]) in splitted_sentence))) or (
                            (splitted_aspects[1] in splitted_sentence) or (
                            (inflection.pluralize(splitted_aspects[1]) in splitted_sentence))):

                        neg = when_there_is_none_aspect_opinion(sentnce, aspect_opnions, review)
                        if len(neg) > 0:
                            neg_lst = neg_lst + neg

                        elif len(r_wrds) > 2:
                            Tgrams = ngrams(r_wrds, 3)
                            for a, b, c in Tgrams:
                                if (a == splitted_aspects[0]) or (a == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(b, c, aspect_opnions)) > 0:
                                        check_for_trigram(b, c, aspect_opnions)
                                elif (b == splitted_aspects[0]) or (b == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(a, c, aspect_opnions)) > 0:
                                        check_for_trigram(a, c, aspect_opnions)
                                elif (c == splitted_aspects[0]) or (c == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(a, b, aspect_opnions)) > 0:
                                        check_for_trigram(a, b, aspect_opnions)
                                if (a == splitted_aspects[0]) or (a == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(c, b, aspect_opnions)) > 0:
                                        check_for_trigram(c, b, aspect_opnions)
                                if (b == splitted_aspects[0]) or (b == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(c, a, aspect_opnions)) > 0:
                                        check_for_trigram(c, a, aspect_opnions)
                                if (c == splitted_aspects[0]) or (c == inflection.pluralize(splitted_aspects[0])):
                                    if len(check_for_trigram(b, a, aspect_opnions)) > 0:
                                        check_for_trigram(b, a, aspect_opnions)
                                if (a == splitted_aspects[1]) or (a == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(b, c, aspect_opnions)) > 0:
                                        check_for_trigram(b, c, aspect_opnions)
                                elif (b == splitted_aspects[1]) or (b == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(a, c, aspect_opnions)) > 0:
                                        check_for_trigram(a, c, aspect_opnions)
                                elif (c == splitted_aspects[1]) or (c == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(a, b, aspect_opnions)) > 0:
                                        check_for_trigram(a, b, aspect_opnions)
                                if (a == splitted_aspects[1]) or (a == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(c, b, aspect_opnions)) > 0:
                                        check_for_trigram(c, b, aspect_opnions)
                                if (b == splitted_aspects[1]) or (b == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(c, a, aspect_opnions)) > 0:
                                        check_for_trigram(c, a, aspect_opnions)
                                if (c == splitted_aspects[1]) or (c == inflection.pluralize(splitted_aspects[1])):
                                    if len(check_for_trigram(b, a, aspect_opnions)) > 0:
                                        check_for_trigram(b, a, aspect_opnions)

                        if len(aspect_opnions) == 1:
                            neg, pos = when_there_is_one_aspect_opinion(sentnce, aspect_opnions, review)
                            if len(neg) > 0:
                                neg_lst = neg_lst + neg
                            elif len(pos) > 0:
                                pos_lst = pos_lst + pos
                        elif len(aspect_opnions) > 1:
                            counter_negatives, counter_positives = counter_negatives_and_positives_opinion_words(
                                aspect_opnions, splitted_sentence)
                            if counter_negatives > counter_positives:

                                neg_lst.append(rev)

                            elif counter_negatives < counter_positives:

                                pos_lst.append(rev)
                            elif counter_negatives == counter_positives:
                                pos_lst.append(rev)
                                neg_lst.append(rev)
                        elif (len(aspect_opnions) == 0) and (sentnce != "") and (
                                (splitted_aspects[0] in splitted_sentence) or (
                                splitted_aspects[1] in splitted_sentence)):
                            neg = when_there_is_none_aspect_opinion(sentnce, aspect_opnions, review)
                            if len(neg) > 0:
                                neg_lst = neg_lst + neg
                if len(neg_lst) > len(pos_lst):
                    positive_and_negative_reviews_for_aspect.Neg_reviews.append(rev)
                elif len(neg_lst) < len(pos_lst):
                    positive_and_negative_reviews_for_aspect.Pos_reviews.append(rev)
        aspect_phrase = phrase_aspect.rstrip()
        aspect_table[aspect_phrase] = positive_and_negative_reviews_for_aspect
    return aspect_table
