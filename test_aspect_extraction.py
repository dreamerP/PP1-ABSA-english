from nltk.corpus import wordnet
from utils_for_testing import read_all_the_clasified_reviews
from utils_general import read_unlabeled_reviews, check_if_has_numbers

'''
not_related_phrases = []
reviews = read_unlabeled_reviews()
related_phrases = []
related_nouns = ['camera']

true_aspect_and_it_reviews_dict = read_all_the_clasified_reviews()
for noun in noun_aspects_bf:
    try:
        if not hasNumber(noun):

            cb1 = wordnet.synset(noun.strip() + ".n.01")
            ib1 = wordnet.synset(str("camera") + ".n.01")
            similarity = cb1.wup_similarity(ib1)
            """
            similarity = owa_agregation_operator(noun)"""
            print(similarity)
            if similarity >= 0.20:
                print('noun ', noun, '  similarity ', similarity)
                related_nouns.append(noun)
    except:
        related_nouns.append(noun)

'''


def get_accuracy_prec_rec(related_nouns, true_aspect_and_it_reviews_dict, not_reltd):
    Tp = 0
    Tn = 0
    Fp = 0
    Fn = 0
    accuracy = 0
    recall = 0
    precision = 0
    f1 = 0
    for aspect in related_nouns.keys():
        if aspect in true_aspect_and_it_reviews_dict.keys():
            Tp = Tp + 1
            print('True positive', aspect)
        else:
            print('Fp', aspect)
            Fp = Fp + 1
    for not_r in not_reltd:
        if not_r not in true_aspect_and_it_reviews_dict.keys():
            print('True negative', not_r)
            Tn = Tn + 1
        else:
            print('Fn', not_r)
            Fn = Fn + 1

    if (Tp + Fp) == 0:
        precision = 0
    if (Tp + Fp) != 0:
        precision = (Tp / (Tp + Fp))
    if (Tp + Fn) == 0:
        recall = 0
    if (Tp + Fn) != 0:
        recall = (Tp / (Tp + Fn))
    if (Tp + Fp + Tn + Fn) != 0:
        accuracy = (Tp + Tn) / (Tp + Fp + Tn + Fn)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * ((recall * precision) / (precision + recall))

    return accuracy, recall, precision, f1
