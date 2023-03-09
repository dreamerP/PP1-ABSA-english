import inflection
from nltk import ngrams
from utils_for_sentiment_clasification import get_orientation, neg_sentiments, negativeWordSet, pos_sentiments, \
    check_for_bigram, check_for_trigram, when_there_is_one_aspect_opinion, \
    counter_negatives_and_positives_opinion_words, if_aspect_opinions_is_empty, get_polarity_using_vader
from utils_general import remove_stop_words, clean_the_review
from tqdm import tqdm


def clasify_phrases_according_to_it_related_synonymns(reviews, phrase_aspect_plus_it_clasified_reviews_table,
                                                      table_phrases_and_it_related_words):
    """
     Esta funcion busca los sinonimos de los noun phrases y actualiza las reseñas
    :param reviews: reseñas
    :param phrase_aspect_plus_it_clasified_reviews_table:
    :param table_phrases_and_it_related_words:
    :return: diccionario con aspecto mas reseñas actualizado con los sinónimos
    """
    for aspect in tqdm(table_phrases_and_it_related_words, 'Buscando sinónimos de las frases sustantivas en las '
                                                           'opiniones y clasificando polaridad'):

        as_synm_lst = table_phrases_and_it_related_words[aspect].synm.split()
        as_synm = table_phrases_and_it_related_words[aspect].synm
        for review in reviews:
            aspect_opnions = []
            exst_empty = False
            sentences = review.split(",")
            for sent in sentences:
                aspect = ""
                equaly = False
                found = False
                sent = sent.split(" ")
                sent = clean_the_review(sent).lower()
                sent = remove_stop_words(sent)
                if len(as_synm_lst) == 1:
                    if (as_synm in sent) or (inflection.pluralize(as_synm) in sent):
                        if len(sent) == 2:
                            bigrams = ngrams(sent, 2)
                            for i, j in bigrams:
                                if (i == as_synm) or (i == inflection.pluralize(as_synm)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == as_synm) or (j == inflection.pluralize(as_synm)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif len(sent) > 2:
                            Tgrams = ngrams(sent, 3)
                            for a, b, c in Tgrams:
                                if (a == as_synm) or (a == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(b, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (b == as_synm) or (b == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(a, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (c == as_synm) or (c == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(a, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (a == as_synm) or (a == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(c, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (b == as_synm) or (b == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(c, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (c == as_synm) or (c == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(b, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result

                elif len(as_synm_lst) >= 2:
                    two = True
                    if (as_synm in sent) or (inflection.pluralize(as_synm) in sent):
                        found = True
                    if (found == True) and (two == True):
                        if (len(as_synm_lst) == 2):
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(sent, 3)
                            for a, b, c in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1])) or (
                                        (a == pwrds[0]) and (b == pwrds[1])):
                                    if check_for_bigram(c, aspect_opnions):
                                        aspect_opnions.append(c)
                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1])) or (
                                        (b == pwrds[0]) and (c == pwrds[1])):
                                    if check_for_bigram(a, aspect_opnions):
                                        aspect_opnions.append(a)
                        elif (len(as_synm_lst) == 3):
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(sent, 4)
                            for a, b, c, d in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2])) or (
                                        (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2])):
                                    if check_for_bigram(d, aspect_opnions):
                                        aspect_opnions.append(d)

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
                                    if check_for_bigram(a, aspect_opnions):
                                        aspect_opnions.append(a)

                        elif (len(as_synm_lst) == 4):
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(sent, 5)
                            for a, b, c, d, e in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2]) and (
                                        d == as_synm_lst[3])) or (
                                        (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2]) and (d == pwrds[3])):
                                    if check_for_bigram(e, aspect_opnions):
                                        aspect_opnions.append(e)

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2]) and (
                                        e == as_synm_lst[3])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2]) and (e == pwrds[3])):
                                    if check_for_bigram(a, aspect_opnions):
                                        aspect_opnions.append(a)
                        elif (len(as_synm_lst) == 5):
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(sent, 6)
                            for a, b, c, d, e, f in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2]) and (
                                        d == as_synm_lst[3]) and (e == as_synm_lst[4])) or (
                                        (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2]) and (
                                        d == pwrds[3]) and (e == pwrds[4])):
                                    if check_for_bigram(f, aspect_opnions):
                                        aspect_opnions.append(f)

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2]) and (
                                        e == as_synm_lst[3]) and (f == as_synm_lst[4])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2]) and (
                                        e == pwrds[3]) and (f == pwrds[4])):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(a)

                    elif (found == False) and (two == True):

                        for s in as_synm_lst:

                            if (s == aspect):
                                equaly = True

                            if (s in sent) or ((inflection.pluralize(s)) in sent) and (equaly == False):
                                if (len(sent) == 2):
                                    bigrams = ngrams(sent, 2)
                                    for i, j in bigrams:
                                        if (i == s) or (i == inflection.pluralize(s)):
                                            if check_for_bigram(j, aspect_opnions):
                                                aspect_opnions.append(j)
                                        elif (j == s) or (j == inflection.pluralize(s)):
                                            if check_for_bigram(i, aspect_opnions):
                                                aspect_opnions.append(i)
                                elif len(sent) > 2:
                                    Tgrams = ngrams(sent, 3)
                                    for a, b, c in Tgrams:
                                        if (a == s) or (a == inflection.pluralize(s)):
                                            result = check_for_trigram(b, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (b == s) or (b == inflection.pluralize(s)):
                                            result = check_for_trigram(a, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (c == s) or (c == inflection.pluralize(s)):
                                            result = check_for_trigram(a, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (a == s) or (a == inflection.pluralize(s)):
                                            result = check_for_trigram(c, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (b == s) or (b == inflection.pluralize(s)):
                                            result = check_for_trigram(c, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (c == s) or (c == inflection.pluralize(s)):
                                            result = check_for_trigram(b, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                if len(aspect_opnions) == 1:
                    negative_rev, pos_rev = when_there_is_one_aspect_opinion(sent, aspect_opnions, review)
                    if len(pos_rev) > 0 and pos_rev[0] not in phrase_aspect_plus_it_clasified_reviews_table[
                        aspect].Pos_reviews:
                        phrase_aspect_plus_it_clasified_reviews_table[aspect].Pos_reviews.append(review)
                    if len(negative_rev) > 0 and negative_rev[0] not in phrase_aspect_plus_it_clasified_reviews_table[
                        aspect].Neg_reviews:
                        phrase_aspect_plus_it_clasified_reviews_table[aspect].Neg_reviews.append(
                            review)
                elif len(aspect_opnions) > 1:
                    c_neg, c_pos = counter_negatives_and_positives_opinion_words(aspect_opnions, sent)
                    if c_neg > c_pos:

                        if review not in phrase_aspect_plus_it_clasified_reviews_table[aspect].Neg_reviews:
                            phrase_aspect_plus_it_clasified_reviews_table[aspect].Neg_reviews.append(review)

                    elif c_neg < c_pos:

                        if review not in phrase_aspect_plus_it_clasified_reviews_table[aspect].Pos_reviews:
                            phrase_aspect_plus_it_clasified_reviews_table[aspect].Pos_reviews.append(review)
                if len(as_synm_lst) == 1:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            as_synm in sent or (inflection.pluralize(as_synm) in sent)):
                        exst_empty = True
                elif len(as_synm_lst) == 2:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (as_synm in sent) or (inflection.pluralize(as_synm) in sent)) or (
                            as_synm_lst[0] in sent) or (inflection.pluralize(as_synm_lst[0]) in sent) or (
                            as_synm_lst[1] in sent) or (inflection.pluralize(as_synm_lst[1]) in sent):
                        exst_empty = True
                elif (len(as_synm_lst) == 3):
                    if (len(aspect_opnions) == 0) and (len(sent) > 0) and (
                            (as_synm in sent) or (inflection.pluralize(as_synm) in sent)) or (
                            as_synm_lst[0] in sent) or (inflection.pluralize(as_synm_lst[0]) in sent) or (
                            as_synm_lst[1] in sent) or (inflection.pluralize(as_synm_lst[1]) in sent) or (
                            as_synm_lst[2] in sent) or (inflection.pluralize(as_synm_lst[2]) in sent):
                        exst_empty = True
                elif (len(as_synm_lst) == 4):
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (as_synm in sent) or (inflection.pluralize(as_synm) in sent)) or (
                            as_synm_lst[0] in sent) or (inflection.pluralize(as_synm_lst[0]) in sent) or (
                            as_synm_lst[1] in sent) or (inflection.pluralize(as_synm_lst[1]) in sent) or (
                            as_synm_lst[2] in sent) or (inflection.pluralize(as_synm_lst[2]) in sent) or (
                            as_synm_lst[3] in sent) or (inflection.pluralize(as_synm_lst[3]) in sent):
                        exst_empty = True

                elif len(as_synm_lst) == 5:
                    if (len(aspect_opnions) == 0) and (len(sent) > 0) and (
                            (as_synm in sent) or (inflection.pluralize(as_synm) in sent)) or (
                            as_synm_lst[0] in sent) or (inflection.pluralize(as_synm_lst[0]) in sent) or (
                            as_synm_lst[1] in sent) or (inflection.pluralize(as_synm_lst[1]) in sent) or (
                            as_synm_lst[2] in sent) or (inflection.pluralize(as_synm_lst[2]) in sent) or (
                            as_synm_lst[3] in sent) or (inflection.pluralize(as_synm_lst[3]) in sent) or (
                            as_synm_lst[4] in sent) or (inflection.pluralize(as_synm_lst[4]) in sent):
                        exst_empty = True

                if exst_empty:
                    neg_rev = if_aspect_opinions_is_empty(sent, as_synm_lst, as_synm_lst, review)
                    if len(neg_rev) > 0 and neg_rev[0] not in phrase_aspect_plus_it_clasified_reviews_table[
                        aspect].Neg_reviews:
                        phrase_aspect_plus_it_clasified_reviews_table[aspect].Neg_reviews.append(review)

    return phrase_aspect_plus_it_clasified_reviews_table


def search_phrase_upers(reviews, aspect_with_classified_reviews, noun_phrase_and_related_words):
    """
    Esta funcion se utiliza para clasificar la
    :param reviews:
    :param aspect_with_classified_reviews:
    :param noun_phrase_and_related_words:
    :return:
    """
    for aspect in tqdm(noun_phrase_and_related_words,
                       desc='Buscando hiperónimos de las frases sustantivas en las opiniones '
                            'y clasificando'):
        upper_word_lst = noun_phrase_and_related_words[aspect].upper.split()
        upper_word = noun_phrase_and_related_words[aspect].upper
        for review in reviews:
            aspect_opnions = []
            exst_empty = False
            sentences = review.split(",")
            for sent in sentences:
                equaly = False
                found = False
                sent = clean_the_review(sent).lower()
                sent = remove_stop_words(sent)
                if len(upper_word_lst) == 1:
                    if (upper_word in sent) or (inflection.pluralize(upper_word) in sent):
                        if len(sent) == 2:
                            bigrams = ngrams(sent, 2)
                            for i, j in bigrams:
                                if (i == upper_word) or (i == inflection.pluralize(upper_word)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == upper_word) or (j == inflection.pluralize(upper_word)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif len(sent) > 2:
                            Tgrams = ngrams(sent, 3)
                            for a, b, c in Tgrams:
                                if (a == upper_word) or (a == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(b, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (b == upper_word) or (b == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(a, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (c == upper_word) or (c == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(a, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (a == upper_word) or (a == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(c, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (b == upper_word) or (b == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(c, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (c == upper_word) or (c == inflection.pluralize(upper_word)):
                                    result = check_for_trigram(b, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result

                elif len(upper_word_lst) >= 2:
                    two = True
                    if (upper_word in sent) or (inflection.pluralize(upper_word) in sent):
                        found = True
                    if (found == True) and (two == True):
                        if len(sent) > 2:
                            if len(upper_word_lst) == 2:
                                plural = inflection.pluralize(upper_word)
                                pwrds = plural.split()
                                Tgrams = ngrams(sent, 3)
                                for a, b, c in Tgrams:
                                    if ((a == upper_word_lst[0]) and (b == upper_word_lst[1])) or (
                                            (a == pwrds[0]) and (b == pwrds[1])):
                                        if check_for_bigram(c, aspect_opnions):
                                            aspect_opnions.append(c)
                                    elif ((b == upper_word_lst[0]) and (c == upper_word_lst[1])) or (
                                            (b == pwrds[0]) and (c == pwrds[1])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                            elif len(upper_word_lst) == 3:
                                plural = inflection.pluralize(upper_word)
                                pwrds = plural.split()
                                Tgrams = ngrams(sent, 4)
                                for a, b, c, d in Tgrams:
                                    if ((a == upper_word_lst[0]) and (b == upper_word_lst[1]) and (
                                            c == upper_word_lst[2])) or (
                                            (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2])):
                                        if check_for_bigram(d, aspect_opnions):
                                            aspect_opnions.append(d)

                                    elif ((b == upper_word_lst[0]) and (c == upper_word_lst[1]) and (
                                            d == upper_word_lst[2])) or (
                                            (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                            elif len(upper_word_lst) == 4:
                                plural = inflection.pluralize(upper_word)
                                pwrds = plural.split()
                                Tgrams = ngrams(sent, 5)
                                for a, b, c, d, e in Tgrams:
                                    if ((a == upper_word_lst[0]) and (b == upper_word_lst[1]) and (
                                            c == upper_word_lst[2]) and (
                                                d == upper_word_lst[3])) or (
                                            (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2]) and (
                                            d == pwrds[3])):
                                        if check_for_bigram(e, aspect_opnions):
                                            aspect_opnions.append(e)

                                    elif ((b == upper_word_lst[0]) and (c == upper_word_lst[1]) and (
                                            d == upper_word_lst[2]) and (e == upper_word_lst[3])) or (
                                            (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2]) and (
                                            e == pwrds[3])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                    elif (found == False) and (two == True):

                        for upper in upper_word_lst:

                            if upper == aspect:
                                equaly = True

                            if (upper in sent) or ((inflection.pluralize(upper)) in sent) and (equaly == False):
                                if len(sent) == 2:
                                    bigrams = ngrams(sent, 2)
                                    for i, j in bigrams:
                                        if (i == upper) or (i == inflection.pluralize(upper)):
                                            if check_for_bigram(j, aspect_opnions):
                                                aspect_opnions.append(j)
                                        elif (j == upper) or (j == inflection.pluralize(upper)):
                                            if check_for_bigram(i, aspect_opnions):
                                                aspect_opnions.append(i)
                                elif len(sent) > 2:
                                    Tgrams = ngrams(sent, 3)
                                    for a, b, c in Tgrams:
                                        if (a == upper) or (a == inflection.pluralize(upper)):
                                            result = check_for_trigram(b, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (b == upper) or (b == inflection.pluralize(upper)):
                                            result = check_for_trigram(a, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (c == upper) or (c == inflection.pluralize(upper)):
                                            result = check_for_trigram(a, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (a == upper) or (a == inflection.pluralize(upper)):
                                            result = check_for_trigram(c, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (b == upper) or (b == inflection.pluralize(upper)):
                                            result = check_for_trigram(c, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (c == upper) or (c == inflection.pluralize(upper)):
                                            result = check_for_trigram(b, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                if len(aspect_opnions) == 1:
                    negative_rev, pos_rev = when_there_is_one_aspect_opinion(sent, aspect_opnions, review)
                    if len(pos_rev) > 0 and pos_rev[0] not in aspect_with_classified_reviews[aspect].Pos_reviews:
                        aspect_with_classified_reviews[aspect].Pos_reviews.append(review)
                    if len(negative_rev) > 0 and negative_rev[0] not in aspect_with_classified_reviews[
                        aspect].Neg_reviews:
                        aspect_with_classified_reviews[aspect].Neg_reviews.append(review)
                elif len(aspect_opnions) > 1:
                    c_neg, c_pos = counter_negatives_and_positives_opinion_words(aspect_opnions, sent)
                    if c_neg > c_pos:

                        if review not in aspect_with_classified_reviews[aspect].Neg_reviews:
                            aspect_with_classified_reviews[aspect].Neg_reviews.append(review)

                    elif c_neg < c_pos:

                        if review not in aspect_with_classified_reviews[aspect].Pos_reviews:
                            aspect_with_classified_reviews[aspect].Pos_reviews.append(review)
                if len(upper_word_lst) == 1:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (upper_word in sent) or (inflection.pluralize(upper_word) in sent)):
                        exst_empty = True
                elif len(upper_word_lst) == 2:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (upper_word in sent) or (inflection.pluralize(upper_word) in sent)) or (
                            upper_word_lst[0] in sent) or (inflection.pluralize(upper_word_lst[0]) in sent) or (
                            upper_word_lst[1] in sent) or (inflection.pluralize(upper_word_lst[1]) in sent):
                        exst_empty = True
                elif len(upper_word_lst) == 3:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (upper_word in sent) or (inflection.pluralize(upper_word) in sent)) or (
                            upper_word_lst[0] in sent) or (inflection.pluralize(upper_word_lst[0]) in sent) or (
                            upper_word_lst[1] in sent) or (inflection.pluralize(upper_word_lst[1]) in sent) or (
                            upper_word_lst[2] in sent) or (inflection.pluralize(upper_word_lst[2]) in sent):
                        exst_empty = True
                elif len(upper_word_lst) == 4:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (upper_word in sent) or (inflection.pluralize(upper_word) in sent)) or (
                            upper_word_lst[0] in sent) or (inflection.pluralize(upper_word_lst[0]) in sent) or (
                            upper_word_lst[1] in sent) or (inflection.pluralize(upper_word_lst[1]) in sent) or (
                            upper_word_lst[2] in sent) or (inflection.pluralize(upper_word_lst[2]) in sent) or (
                            upper_word_lst[3] in sent) or (inflection.pluralize(upper_word_lst[3]) in sent):
                        exst_empty = True
                if (exst_empty == True):
                    neg_rev = if_aspect_opinions_is_empty(sent, upper_word_lst, upper_word, review)
                    if len(neg_rev) > 0 and neg_rev[0] not in aspect_with_classified_reviews[
                        aspect].Neg_reviews:
                        aspect_with_classified_reviews[aspect].Neg_reviews.append(review)
    return aspect_with_classified_reviews


def search_for_phrase_aspects_lowers(reviews, aspect_and_reviews, aspect_and_related_words):
    """
    Esta funcion se utiliza para buscar los hiponimos de las noun phrases que se encuetren presentes en
    las reseñas
    :param reviews:
    :param aspect_and_reviews:
    :param aspect_and_related_words:
    :return:
    """
    found_neg = False
    for aspect in tqdm(aspect_and_related_words, desc='Buscando hiponimos de las frases sustantivas en las opiniones y '
                                                      'clasificando polaridad'):

        as_synm_lst = (aspect_and_related_words[aspect].lwr).split()
        as_synm = inflection.singularize((aspect_and_related_words[aspect].lwr))
        for r in reviews:

            rev = r
            aspect_opnions = []
            exst_empty = False
            rev_wrds = rev.split()
            sentences = rev.split(",")
            for sent in sentences:

                equaly = False
                found = False
                one = False
                two = False
                wreds = sent.split(" ")
                revv = clean_the_review(sent).lower()
                r_wrds = remove_stop_words(revv)
                if (len(as_synm_lst) == 1):
                    if (as_synm in r_wrds) or (inflection.pluralize(as_synm) in r_wrds):
                        if len(r_wrds) == 2:
                            bigrams = ngrams(r_wrds, 2)
                            for i, j in bigrams:
                                if (i == as_synm) or (i == inflection.pluralize(as_synm)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == as_synm) or (j == inflection.pluralize(as_synm)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif len(r_wrds) > 2:
                            Tgrams = ngrams(r_wrds, 3)

                            for a, b, c in Tgrams:
                                if (a == as_synm) or (a == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(b, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (b == as_synm) or (b == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(a, c, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                elif (c == as_synm) or (c == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(a, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (a == as_synm) or (a == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(c, b, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (b == as_synm) or (b == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(c, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result
                                if (c == as_synm) or (c == inflection.pluralize(as_synm)):
                                    result = check_for_trigram(b, a, aspect_opnions)
                                    if len(result) > 0:
                                        aspect_opnions = aspect_opnions + result

                elif (len(as_synm_lst) >= 2):
                    two = True
                    if (revv.find(as_synm) != -1) or (revv.find(inflection.pluralize(as_synm)) != -1):
                        found = True
                    if (found == True) and (two == True):
                        if (len(r_wrds) > 2):
                            if (len(as_synm_lst) == 2):
                                plural = inflection.pluralize(as_synm)
                                pwrds = plural.split()
                                Tgrams = ngrams(r_wrds, 3)
                                for a, b, c in Tgrams:
                                    if ((a == as_synm_lst[0]) and (b == as_synm_lst[1])) or (
                                            (a == pwrds[0]) and (b == pwrds[1])):
                                        if check_for_bigram(c, aspect_opnions):
                                            aspect_opnions.append(c)
                                    elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1])) or (
                                            (b == pwrds[0]) and (c == pwrds[1])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                            elif (len(as_synm_lst) == 3):
                                plural = inflection.pluralize(as_synm)
                                pwrds = plural.split()
                                Tgrams = ngrams(r_wrds, 4)
                                for a, b, c, d in Tgrams:
                                    if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2])) or (
                                            (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2])):
                                        if check_for_bigram(d, aspect_opnions):
                                            aspect_opnions.append(d)
                                    elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (
                                            d == as_synm_lst[2])) or (
                                            (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                            elif (len(as_synm_lst) == 4):
                                plural = inflection.pluralize(as_synm)
                                pwrds = plural.split()
                                Tgrams = ngrams(r_wrds, 5)
                                for a, b, c, d, e in Tgrams:
                                    if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2]) and (
                                            d == as_synm_lst[3])) or (
                                            (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2]) and (
                                            d == pwrds[3])):
                                        if check_for_bigram(e, aspect_opnions):
                                            aspect_opnions.append(e)
                                    elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (
                                            d == as_synm_lst[2]) and (e == as_synm_lst[3])) or (
                                            (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2]) and (
                                            e == pwrds[3])):
                                        if check_for_bigram(a, aspect_opnions):
                                            aspect_opnions.append(a)
                    elif (found == False) and (two == True):

                        for s in as_synm_lst:

                            if (s == aspect):
                                equaly = True

                            if (s in r_wrds) or ((inflection.pluralize(s)) in r_wrds) and (equaly == False):
                                if (len(r_wrds) == 2):
                                    bigrams = ngrams(r_wrds, 2)
                                    for i, j in bigrams:
                                        if (i == s) or (i == inflection.pluralize(s)):
                                            if check_for_bigram(j, aspect_opnions):
                                                aspect_opnions.append(j)
                                        elif (j == s) or (j == inflection.pluralize(s)):
                                            if check_for_bigram(i, aspect_opnions):
                                                aspect_opnions.append(i)
                                elif (len(r_wrds) > 2):
                                    Tgrams = ngrams(r_wrds, 3)
                                    for a, b, c in Tgrams:
                                        if (a == s) or (a == inflection.pluralize(s)):
                                            result = check_for_trigram(b, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (b == s) or (b == inflection.pluralize(s)):
                                            result = check_for_trigram(a, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (c == s) or (c == inflection.pluralize(s)):
                                            result = check_for_trigram(a, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (a == s) or (a == inflection.pluralize(s)):
                                            result = check_for_trigram(c, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (b == s) or (b == inflection.pluralize(s)):
                                            result = check_for_trigram(c, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (c == s) or (c == inflection.pluralize(s)):
                                            result = check_for_trigram(b, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                if (len(aspect_opnions) == 1):
                    rev1 = sent.lower()
                    rev_words = rev1.split()
                    polarity = get_polarity_using_vader(aspect_opnions[0])
                    postv = False
                    negtv = False
                    if (polarity == '+') or '=':
                        for p in pos_sentiments:
                            if (p.strip() in rev_words):
                                postv = True
                                break
                    elif (polarity == '-') or polarity == '=':
                        for n in neg_sentiments:
                            if (n.strip() in rev_words):
                                negtv = True
                                break
                    exist = False
                    if ((polarity == '+') and (postv == True)) or ((polarity != '+') and (postv == True)) or (
                            (polarity == '=') and (postv == True)):

                        for i in rev_words:
                            for j in negativeWordSet:
                                if (i == j):
                                    found_neg = True
                                    neg_word = j
                                    exist = True
                                    break
                                else:
                                    found_neg = False
                            if (exist == True):
                                break
                        if (found_neg == False):

                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                aspect_and_reviews[aspect].Pos_reviews.append(rev)


                        elif (found_neg == True):
                            if (aspect_opnions[0] in rev_words):
                                if (rev_words.index(neg_word) < rev_words.index(aspect_opnions[0])):
                                    if (neg_word == "not"):
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                        else:

                                            if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                    else:

                                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                elif rev_words.index(neg_word) > rev_words.index(aspect_opnions[0]):

                                    if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                        aspect_and_reviews[aspect].Pos_reviews.append(rev)

                    elif ((polarity == '-') and (negtv == True)) or ((polarity != '-') and (negtv == True)) or (
                            (polarity == '=') and (negtv == True)):

                        for i in rev_words:
                            for j in negativeWordSet:
                                if (i == j):
                                    found_neg = True
                                    neg_word = j
                                    exist = True
                                    break
                                else:
                                    found_neg = False
                            if exist == True:
                                break
                        if (found_neg == False):

                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (found_neg == True):
                            if (aspect_opnions[0] in rev_words):
                                if rev_words.index(neg_word) < rev_words.index(aspect_opnions[0]):
                                    if (neg_word == "not"):
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                    else:

                                        if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                            aspect_and_reviews[aspect].Pos_reviews.append(rev)


                                elif (rev_words.index(neg_word) > rev_words.index(aspect_opnions[0])):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (polarity == '=') and (postv == True):

                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)
                            break
                    elif (polarity == '=') and (negtv == True):

                        if rev not in aspect_and_reviews[aspect].Neg_reviews:
                            aspect_and_reviews[aspect].Neg_reviews.append(rev)
                            break

                elif (len(aspect_opnions) > 1):
                    c_pos = 0
                    c_neg = 0
                    rev1 = sent.lower()
                    rev_words = rev1.split()
                    for op in aspect_opnions:
                        pol = get_polarity_using_vader(op)
                        postv = False
                        negtv = False
                        if (pol == '+') or pol == '=':
                            for p in pos_sentiments:
                                if (p.strip() in rev_words):
                                    postv = True
                                    break
                        elif (pol == '-') or pol == '=':
                            for n in neg_sentiments:
                                if (n.strip() in rev_words):
                                    negtv = True
                                    break

                        exist = False
                        if ((pol == '+') and (postv == True)) or ((pol != '+') and (postv == True)) or (
                                (pol == '=') and (postv == True)) or ((pol == '+') and (postv == False)):

                            for i in rev_words:
                                for j in negativeWordSet:
                                    if (i == j):
                                        found_neg = True
                                        neg_word = j
                                        exist = True
                                        break
                                    else:
                                        found_neg = False
                                if exist == True:
                                    break
                            if found_neg == False:
                                c_pos += 1
                            elif (found_neg == True):
                                if op in rev_words:
                                    if rev_words.index(neg_word) < rev_words.index(op):
                                        if neg_word == "not":
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_pos += 1
                                            else:
                                                c_neg += 1
                                        else:
                                            c_neg += 1
                                    elif rev_words.index(neg_word) > rev_words.index(op):
                                        c_pos += 1
                        elif ((pol == '-') and (negtv == True)) or ((pol != '-') and (negtv == True)) or (
                                (pol == '=') and (negtv == True)) or ((pol == '-') and (negtv == False)):

                            for i in rev_words:
                                for j in negativeWordSet:
                                    if (i == j):
                                        found_neg = True
                                        neg_word = j
                                        exist = True
                                        break
                                    else:
                                        found_neg = False
                                if (exist == True):
                                    break
                            if found_neg == False:
                                c_neg += 1
                            elif found_neg == True:
                                if (op in rev_words):
                                    if rev_words.index(neg_word) < rev_words.index(op):
                                        if neg_word == "not":
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_neg += 1
                                            else:
                                                c_pos += 1
                                        else:
                                            c_pos += 1
                                    elif (rev_words.index(neg_word) > rev_words.index(op)):
                                        c_neg += 1
                    if (c_neg > c_pos):

                        if rev not in aspect_and_reviews[aspect].Neg_reviews:
                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif c_neg < c_pos:
                        if rev not in aspect_and_reviews[aspect].Pos_reviews:
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)

                sentos = sent.split()

                if len(as_synm_lst) == 1:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)):
                        exst_empty = True
                elif len(as_synm_lst) == 2:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)) or (
                            as_synm_lst[0] in sentos) or (inflection.pluralize(as_synm_lst[0]) in sentos) or (
                            as_synm_lst[1] in sentos) or (inflection.pluralize(as_synm_lst[1]) in sentos):
                        exst_empty = True
                elif len(as_synm_lst) == 3:
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)) or (
                            as_synm_lst[0] in sentos) or (inflection.pluralize(as_synm_lst[0]) in sentos) or (
                            as_synm_lst[1] in sentos) or (inflection.pluralize(as_synm_lst[1]) in sentos) or (
                            as_synm_lst[2] in sentos) or (inflection.pluralize(as_synm_lst[2]) in sentos):
                        exst_empty = True
                if exst_empty == True:
                    rev1 = rev.lower()
                    rev_words = rev1.split()
                    exist = False
                    for i in rev_words:
                        for j in negativeWordSet:
                            if (i == j):
                                found_neg_p = True
                                neg_word = j
                                exist = True
                                break
                            else:
                                found_neg_p = False
                        if (exist == True):
                            break
                    if (found_neg_p == True):
                        if len(as_synm_lst) == 1:
                            if as_synm in rev_words:

                                if rev_words.index(neg_word) < rev_words.index(as_synm):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if inflection.pluralize(as_synm) in rev_words:
                                asspkt = inflection.singularize(as_synm)
                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(asspkt)):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (len(as_synm_lst) > 1):

                            if (sent.find(as_synm) != -1):

                                if (rev_words.index(neg_word) < rev_words.index(as_synm_lst[0])):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if sent.find(inflection.pluralize(as_synm)) != -1:
                                asspkt = inflection.singularize(as_synm)
                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(as_synm_lst[0])):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

    return aspect_and_reviews
