import inflection
from nltk import ngrams
from utils_for_sentiment_clasification import pos_sentiments, neg_sentiments, negativeWordSet, \
    check_for_bigram, check_for_trigram, get_polarity_using_vader
from utils_general import clean_the_review, remove_stop_words
from tqdm import tqdm


def search_for_synonym_and_update_classified_reviews(reviews, aspect_and_reviews, aspect_and_related_words):
    """
    Esta funcion busca los sinonimos de los aspectos que se encuentren en las reseñas y actualiza el diccionario de
    reseñas clasificadas
    :param reviews: reeñas
    :param aspect_and_reviews: aspectos explicitos con sus reseñas encontradas
    :param aspect_and_related_words: apalbras relacionadas con cada aspecto
    :return: diccionario de reseñas y aspectos actualizado
    """
    found_neg = False

    for aspect in tqdm(aspect_and_related_words, desc='Buscando los sinónimos de los aspectos sustantivos y '
                                                      'clasificando reseñas'):

        as_synm_lst = aspect_and_related_words[aspect].synm.split()
        as_synm = aspect_and_related_words[aspect].synm
        for r in reviews:
            rev = r
            aspect_opnions = []
            exst_empty = False
            sentences = rev.split(",")
            for sent in sentences:
                equaly = False
                clean_step1 = clean_the_review(sent).lower()
                clean_sentence = remove_stop_words(clean_step1)
                if len(as_synm_lst) == 1:
                    if (as_synm in clean_sentence) or (inflection.pluralize(as_synm) in clean_sentence):
                        if len(clean_sentence) == 2:
                            bigrams = ngrams(clean_sentence, 2)
                            for i, j in bigrams:
                                if (i == as_synm) or (i == inflection.pluralize(as_synm)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == as_synm) or (j == inflection.pluralize(as_synm)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif len(clean_sentence) > 2:
                            Tgrams = ngrams(clean_sentence, 3)
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

                    if (clean_step1.find(as_synm) != -1) or (clean_step1.find(inflection.pluralize(as_synm)) != -1):

                        if len(as_synm_lst) == 2:
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(clean_sentence, 3)
                            for a, b, c in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1])) or (
                                        (a == pwrds[0]) and (b == pwrds[1])):
                                    if check_for_bigram(c, aspect_opnions):
                                        aspect_opnions.append(c)
                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1])) or (
                                        (b == pwrds[0]) and (c == pwrds[1])):
                                    if check_for_bigram(a, aspect_opnions):
                                        aspect_opnions.append(a)
                        elif len(as_synm_lst) == 3:
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(clean_sentence, 4)
                            for a, b, c, d in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2])) or (
                                        (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2])):
                                    if check_for_bigram(d, aspect_opnions):
                                        aspect_opnions.append(d)

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
                                    if check_for_bigram(a, aspect_opnions):
                                        aspect_opnions.append(a)
                    else:

                        for synonym in as_synm_lst:
                            if synonym == aspect:
                                equaly = True

                            if (synonym in clean_sentence) or ((inflection.pluralize(synonym)) in clean_sentence) and (
                                    equaly == False):
                                if len(clean_sentence) == 2:
                                    bigrams = ngrams(clean_sentence, 2)
                                    for i, j in bigrams:
                                        if (i == synonym) or (i == inflection.pluralize(synonym)):
                                            if check_for_bigram(j, aspect_opnions):
                                                aspect_opnions.append(j)
                                        elif (j == synonym) or (j == inflection.pluralize(synonym)):
                                            if check_for_bigram(i, aspect_opnions):
                                                aspect_opnions.append(i)
                                elif len(clean_sentence) > 2:
                                    Tgrams = ngrams(clean_sentence, 3)
                                    for a, b, c in Tgrams:
                                        if (a == synonym) or (a == inflection.pluralize(synonym)):
                                            result = check_for_trigram(b, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (b == synonym) or (b == inflection.pluralize(synonym)):
                                            result = check_for_trigram(a, c, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        elif (c == synonym) or (c == inflection.pluralize(synonym)):
                                            result = check_for_trigram(a, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (a == synonym) or (a == inflection.pluralize(synonym)):
                                            result = check_for_trigram(c, b, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (b == synonym) or (b == inflection.pluralize(synonym)):
                                            result = check_for_trigram(c, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                                        if (c == synonym) or (c == inflection.pluralize(synonym)):
                                            result = check_for_trigram(b, a, aspect_opnions)
                                            if len(result) > 0:
                                                aspect_opnions = aspect_opnions + result
                if len(aspect_opnions) == 1:
                    rev1 = sent.lower()
                    rev_words = rev1.split()
                    polarity = get_polarity_using_vader(aspect_opnions[0])
                    postv = False
                    negtv = False
                    if (polarity == '+') or polarity == '=':
                        for p in pos_sentiments:
                            if p.strip() in rev_words:
                                postv = True
                                break
                    elif (polarity == '-') or polarity == '=':
                        for n in neg_sentiments:
                            if n.strip() in rev_words:
                                negtv = True
                                break
                    exist = False
                    if ((polarity == '+') and (postv == True)) or ((polarity != '+') and (postv == True)) or (
                            (polarity == '=') and (postv == True)):

                        for i in rev_words:
                            for j in negativeWordSet:
                                if i == j:
                                    found_neg = True
                                    neg_word = j
                                    exist = True
                                    break
                                else:
                                    found_neg = False
                            if exist == True:
                                break
                        if found_neg == False:

                            if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                        elif found_neg == True:
                            if aspect_opnions[0] in rev_words:
                                if rev_words.index(neg_word) < rev_words.index(aspect_opnions[0]):
                                    if neg_word == "not":
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
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
                            if (exist == True):
                                break
                        if (found_neg == False):

                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (found_neg == True):
                            if aspect_opnions[0] in rev_words:
                                if (rev_words.index(neg_word) < rev_words.index(aspect_opnions[0])):
                                    if (neg_word == "not"):
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                    else:

                                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                            aspect_and_reviews[aspect].Pos_reviews.append(rev)

                            elif (aspect_opnions[0] in rev_words):
                                if rev_words.index(neg_word) > rev_words.index(aspect_opnions[0]):
                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (polarity == '=') and (postv == True):

                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)
                            break
                    elif (polarity == '=') and (negtv == True):

                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
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
                                if (exist == True):
                                    break
                            if (found_neg == False):
                                c_pos += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if (rev_words.index(neg_word) < rev_words.index(op)):
                                        if (neg_word == "not"):
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_pos += 1
                                            else:
                                                c_neg += 1
                                        else:
                                            c_neg += 1
                                    elif (rev_words.index(neg_word) > rev_words.index(op)):
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
                            if (found_neg == False):
                                c_neg += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if (rev_words.index(neg_word) < rev_words.index(op)):
                                        if (neg_word == "not"):
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_neg += 1
                                            else:
                                                c_pos += 1
                                        else:
                                            c_pos += 1
                                    elif (rev_words.index(neg_word) > rev_words.index(op)):
                                        c_neg += 1
                    if c_neg > c_pos:
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
                            if i == j:
                                found_neg_p = True
                                neg_word = j
                                exist = True
                                break
                            else:
                                found_neg_p = False
                        if exist == True:
                            break
                    if found_neg_p == True:
                        if len(as_synm_lst) == 1:
                            if as_synm in rev_words:

                                if rev_words.index(neg_word) < rev_words.index(as_synm):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if inflection.pluralize(as_synm) in rev_words:
                                asspkt = inflection.singularize(as_synm)
                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(asspkt)):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif len(as_synm_lst) > 1:

                            if sent.find(as_synm) != -1:
                                if as_synm_lst[0] in rev_words:
                                    if rev_words.index(neg_word) < rev_words.index(as_synm_lst[0]):

                                        if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if sent.find(inflection.pluralize(as_synm)) != -1:

                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(as_synm_lst[0])):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

    return aspect_and_reviews


def search_for_uppers_and_update_classified_reviews(reviews, aspect_and_reviews, aspect_and_related_words):
    """
    Esta función se utiliza para buscar los hiperonimos de los aspectos que esten presentes en las reseñas y
    se clasifica la polaridad
    :param reviews:
    :param aspect_and_reviews:
    :param aspect_and_related_words:
    :return:
    """
    found_neg = False

    for aspect in tqdm(aspect_and_related_words, desc='Buscando los hiperónimos de los aspectos sustantivos y '
                                                      'clasificando reseñas'):

        as_synm_lst = aspect_and_related_words[aspect].upper.split()
        as_synm = aspect_and_related_words[aspect].upper
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
                        if (len(r_wrds) == 2):
                            bigrams = ngrams(r_wrds, 2)
                            for i, j in bigrams:
                                if (i == as_synm) or (i == inflection.pluralize(as_synm)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == as_synm) or (j == inflection.pluralize(as_synm)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif (len(r_wrds) > 2):
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

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
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
                    if (polarity == '+') or polarity == '=':
                        for p in pos_sentiments:
                            if (p.strip() in rev_words):
                                postv = True
                                break
                    elif (polarity == '-') or pol == '=':
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

                                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                    else:

                                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                elif (rev_words.index(neg_word) > rev_words.index(aspect_opnions[0])):

                                    if (rev not in aspect_and_reviews[aspect].Pos_reviews):
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
                            if (exist == True):
                                break
                        if (found_neg == False):

                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (found_neg == True):
                            if (aspect_opnions[0] in rev_words):
                                if (rev_words.index(neg_word) < rev_words.index(aspect_opnions[0])):
                                    if (neg_word == "not"):
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                    else:

                                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                            aspect_and_reviews[aspect].Pos_reviews.append(rev)


                                elif (rev_words.index(neg_word) > rev_words.index(aspect_opnions[0])):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (polarity == '=') and (postv == True):

                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)
                            break
                    elif (polarity == '=') and (negtv == True):

                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
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
                                if (exist == True):
                                    break
                            if (found_neg == False):
                                c_pos += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if (rev_words.index(neg_word) < rev_words.index(op)):
                                        if (neg_word == "not"):
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_pos += 1
                                            else:
                                                c_neg += 1
                                        else:
                                            c_neg += 1
                                    elif (rev_words.index(neg_word) > rev_words.index(op)):
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
                            if (found_neg == False):
                                c_neg += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if (rev_words.index(neg_word) < rev_words.index(op)):
                                        if (neg_word == "not"):
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

                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (c_neg < c_pos):

                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)
                sentos = sent.split()
                if (len(as_synm_lst) == 1):
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)):
                        exst_empty = True
                elif (len(as_synm_lst) == 2):
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)) or (
                            as_synm_lst[0] in sentos) or (inflection.pluralize(as_synm_lst[0]) in sentos) or (
                            as_synm_lst[1] in sentos) or (inflection.pluralize(as_synm_lst[1]) in sentos):
                        exst_empty = True
                elif (len(as_synm_lst) == 3):
                    if (len(aspect_opnions) == 0) and (sent != "") and (
                            (sent.find(as_synm) != -1) or (sent.find(inflection.pluralize(as_synm)) != -1)) or (
                            as_synm_lst[0] in sentos) or (inflection.pluralize(as_synm_lst[0]) in sentos) or (
                            as_synm_lst[1] in sentos) or (inflection.pluralize(as_synm_lst[1]) in sentos) or (
                            as_synm_lst[2] in sentos) or (inflection.pluralize(as_synm_lst[2]) in sentos):
                        exst_empty = True
                if (exst_empty == True):
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
                        if (len(as_synm_lst) == 1):
                            if (as_synm in rev_words):

                                if (rev_words.index(neg_word) < rev_words.index(as_synm)):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if (inflection.pluralize(as_synm) in rev_words):
                                asspkt = inflection.singularize(as_synm)
                                if (rev_words.index(neg_word) < rev_words.index(inflection.pluralize(asspkt))):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (len(as_synm_lst) > 1):

                            if (sent.find(as_synm) != -1):

                                if (rev_words.index(neg_word) < rev_words.index(as_synm_lst[0])):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if (sent.find(inflection.pluralize(as_synm)) != -1):
                                if (rev_words.index(neg_word) < rev_words.index(inflection.pluralize(as_synm_lst[0]))):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

    return aspect_and_reviews


def search_for_lowers_of_nouns_and_classify_reviews(reviews, aspect_and_reviews, aspect_and_related_words):
    """
    Esta función se utiliza para buscar los hipónimos de los aspectos en las reseñas y clasificarlos
    :param reviews: reseñas
    :param aspect_and_reviews: aspectos con sus reseñas previamente encontrados
    :param aspect_and_related_words: aspectos con sus palabras mas cercanas
    :return: diccionario de aspecto mas resenas actualizado con los hipónimos encontrados
    """
    found_neg = False

    for aspect in tqdm(aspect_and_related_words, desc='Buscando los hipónimos de los aspectos sustantivos y '
                                                      'clasificando reseñas'):

        as_synm_lst = (aspect_and_related_words[aspect].lwr).split()
        as_synm = (aspect_and_related_words[aspect].lwr)
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
                        if (len(r_wrds) == 2):
                            bigrams = ngrams(r_wrds, 2)
                            for i, j in bigrams:
                                if (i == as_synm) or (i == inflection.pluralize(as_synm)):
                                    if check_for_bigram(j, aspect_opnions):
                                        aspect_opnions.append(j)
                                elif (j == as_synm) or (j == inflection.pluralize(as_synm)):
                                    if check_for_bigram(i, aspect_opnions):
                                        aspect_opnions.append(i)
                        elif (len(r_wrds) > 2):
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
                        elif len(as_synm_lst) == 3:
                            plural = inflection.pluralize(as_synm)
                            pwrds = plural.split()
                            Tgrams = ngrams(r_wrds, 4)
                            for a, b, c, d in Tgrams:
                                if ((a == as_synm_lst[0]) and (b == as_synm_lst[1]) and (c == as_synm_lst[2])) or (
                                        (a == pwrds[0]) and (b == pwrds[1]) and (c == pwrds[2])):
                                    if check_for_bigram(d, aspect_opnions):
                                        aspect_opnions.append(d)

                                elif ((b == as_synm_lst[0]) and (c == as_synm_lst[1]) and (d == as_synm_lst[2])) or (
                                        (b == pwrds[0]) and (c == pwrds[1]) and (d == pwrds[2])):
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
                    if polarity == '+' or polarity == '=':
                        for p in pos_sentiments:
                            if (p.strip() in rev_words):
                                postv = True
                                break
                    elif polarity == '-' or polarity == '=':
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

                                            if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                        else:

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
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
                            if (exist == True):
                                break
                        if (found_neg == False):

                            if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif (found_neg == True):
                            if (aspect_opnions[0] in rev_words):
                                if (rev_words.index(neg_word) < rev_words.index(aspect_opnions[0])):
                                    if (neg_word == "not"):
                                        if ("only" in rev_words) and (
                                                ((rev_words.index("only")) - (rev_words.index("not"))) == 1):

                                            if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                                aspect_and_reviews[aspect].Neg_reviews.append(rev)

                                        else:

                                            if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                                aspect_and_reviews[aspect].Pos_reviews.append(rev)

                                    else:

                                        if rev not in aspect_and_reviews[aspect].Pos_reviews:
                                            aspect_and_reviews[aspect].Pos_reviews.append(rev)


                                elif (rev_words.index(neg_word) > rev_words.index(aspect_opnions[0])):

                                    if (rev not in aspect_and_reviews[aspect].Neg_reviews):
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (polarity == '=') and (postv == True):

                        if (rev not in aspect_and_reviews[aspect].Pos_reviews):
                            aspect_and_reviews[aspect].Pos_reviews.append(rev)
                            break
                    elif (polarity == '=') and (negtv == True):

                        if (rev not in aspect_and_reviews[aspect].Neg_reviews):
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
                                if (exist == True):
                                    break
                            if (found_neg == False):
                                c_pos += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if rev_words.index(neg_word) < rev_words.index(op):
                                        if (neg_word == "not"):
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_pos += 1
                                            else:
                                                c_neg += 1
                                        else:
                                            c_neg += 1
                                    elif (rev_words.index(neg_word) > rev_words.index(op)):
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
                            if (found_neg == False):
                                c_neg += 1
                            elif (found_neg == True):
                                if (op in rev_words):
                                    if (rev_words.index(neg_word) < rev_words.index(op)):
                                        if (neg_word == "not"):
                                            if ("only" in rev_words) and (
                                                    ((rev_words.index("only")) - (rev_words.index("not"))) == 1):
                                                c_neg += 1
                                            else:
                                                c_pos += 1
                                        else:
                                            c_pos += 1
                                    elif rev_words.index(neg_word) > rev_words.index(op):
                                        c_neg += 1
                    if c_neg > c_pos:

                        if rev not in aspect_and_reviews[aspect].Neg_reviews:
                            aspect_and_reviews[aspect].Neg_reviews.append(rev)

                    elif (c_neg < c_pos):

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
                            if i == j:
                                found_neg_p = True
                                neg_word = j
                                exist = True
                                break
                            else:
                                found_neg_p = False
                        if exist == True:
                            break
                    if found_neg_p == True:
                        if len(as_synm_lst) == 1:
                            if as_synm in rev_words:

                                if rev_words.index(neg_word) < rev_words.index(as_synm):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if inflection.pluralize(as_synm) in rev_words:

                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(as_synm)):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                        elif len(as_synm_lst) > 1:

                            if sent.find(as_synm) != -1:

                                if rev_words.index(neg_word) < rev_words.index(as_synm_lst[0]):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

                            if sent.find(inflection.pluralize(as_synm)) != -1:

                                if rev_words.index(neg_word) < rev_words.index(inflection.pluralize(as_synm_lst[0])):

                                    if rev not in aspect_and_reviews[aspect].Neg_reviews:
                                        aspect_and_reviews[aspect].Neg_reviews.append(rev)

    return aspect_and_reviews
