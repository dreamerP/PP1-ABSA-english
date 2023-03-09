from nltk.corpus import wordnet
from aspect_extraction import get_most_frequent_aspects
from polarity_words_detection import classify_polarity_for_noun_aspects, classify_polarity_for_noun_phrases
from sentiment_aspects_clasification import search_for_synonym_and_update_classified_reviews, \
    search_for_uppers_and_update_classified_reviews, search_for_lowers_of_nouns_and_classify_reviews
from sentiment_phrases_clasification import search_phrase_upers, search_for_phrase_aspects_lowers, \
    clasify_phrases_according_to_it_related_synonymns
from utils_for_testing import get_precision_recall_accuracy, calculate_f_measure, avg_acc_rec_per, draw_graph, \
    pie_chart_graf, \
    read_all_the_clasified_reviews, test_aspect_extraction
from utils_general import read_unlabeled_reviews, check_if_has_numbers, get_aspect_table_and_it_related_words
from utils_for_aspect_extraction import get_normalized_vector, zimmerman
from tqdm import tqdm
not_related_aspects = []
related_phrases = []
related_nouns = []
reviews = read_unlabeled_reviews()
noun_aspects, phrase_aspects, phrase_aspects_and_structure_dict, not_related_aspects = get_most_frequent_aspects(
    reviews)


# Identificar de los sustantivos extraídos aquellos que están relacionados con el dominio
for noun in noun_aspects:
    try:
        if not check_if_has_numbers(noun):
            vector = get_normalized_vector(noun)
            similarity = round(zimmerman(vector), 4)
            if similarity >= 0.8:
                related_nouns.append(noun)
            elif noun == 'camera' and noun not in related_nouns:
                related_nouns.append(noun)
    except:
        related_nouns.append(noun)


# Identificar las noun phrases relacionadas con el dominio
for phrase in phrase_aspects:
    splitted_phrase = phrase.split()
    similarity1 = 0
    similarity2 = 0

    try:
        if (check_if_has_numbers(splitted_phrase[0]) == False) and (check_if_has_numbers(splitted_phrase[1]) == False):
            cb1 = wordnet.synset(splitted_phrase[0] + ".n.01")
            ib1 = wordnet.synset(str("camera") + ".n.01")
            similarity1 = ib1.wup_similarity(cb1)
            cb2 = wordnet.synset(splitted_phrase[1] + ".n.01")
            ib2 = wordnet.synset(str("camera") + ".n.01")
            similarity2 = ib2.wup_similarity(cb2)
            if similarity1 > 0.11 or similarity2 > 0.11:
                related_phrases.append(phrase)
    except:

        related_phrases.append(phrase)

# Esta variable guarda un diccionario que contiene como key los aspectos aspect_positive_reviews aspect phrases y el
# value son las reseñas con su clasificación de polaridad extraidos de un set de datos etiquetado
true_aspect_and_it_reviews_dict = read_all_the_clasified_reviews()


print("Los sustantivos frecuentes identificados")
print(noun_aspects+phrase_aspects)

print("Aspectos finales: palabras más relacionadas con el dominio")
print(related_nouns+related_phrases)

aspects = []
recall = []
percison = []
accuracy = []
dictionary = {}
ass_ph = []
ass_n = []
f_measure = []
print('Detectando polaridad...')
product_related_phrases_dict = {}
for related_phrase in related_phrases:
    if related_phrase in phrase_aspects_and_structure_dict:
        product_related_phrases_dict[related_phrase] = phrase_aspects_and_structure_dict[related_phrase]

table_phrases_and_it_related_words, table_aspect_phrases_detail_dict, table_noun_and_related_words = get_aspect_table_and_it_related_words(
    product_related_phrases_dict, related_nouns)

# Proceso de clasificación de sentiemientos para los sustantivos relacionados con el dominio
aspect_plus_it_clasified_reviews_table = classify_polarity_for_noun_aspects(reviews, table_noun_and_related_words)
aspect_and_synonymns_clasified_reviews = search_for_synonym_and_update_classified_reviews(reviews,
                                                                                          aspect_plus_it_clasified_reviews_table,
                                                                                          table_noun_and_related_words)
aspect_and_synonymns_and_uppers_clasified_reviews = search_for_uppers_and_update_classified_reviews(reviews,
                                                                                                    aspect_and_synonymns_clasified_reviews,
                                                                                                    table_noun_and_related_words)
final1 = search_for_lowers_of_nouns_and_classify_reviews(reviews, aspect_and_synonymns_and_uppers_clasified_reviews,
                                                         table_noun_and_related_words)

# Proceso de clasificación de sentiemientos para las noun phrases
phrase_aspect_plus_it_clasified_reviews_table = classify_polarity_for_noun_phrases(reviews, related_phrases)
phrase_aspect_and_it_synonyms_clasified_reviews = clasify_phrases_according_to_it_related_synonymns(reviews,
                                                                                                    phrase_aspect_plus_it_clasified_reviews_table,
                                                                                                    table_phrases_and_it_related_words)
phrase_aspect_and_it_uppers_classified_reviews = search_phrase_upers(reviews,
                                                                     phrase_aspect_and_it_synonyms_clasified_reviews,
                                                                     table_phrases_and_it_related_words)
final2 = search_for_phrase_aspects_lowers(reviews, phrase_aspect_and_it_uppers_classified_reviews,
                                          table_phrases_and_it_related_words)

dictionary = final1.copy()
dictionary.update(final2)

# Calculo de las metricas por aspectos
for aspect in dictionary:
    aspect_positive_reviews = []
    aspect_positive_reviews = set(dictionary[aspect].Pos_reviews)
    aspect_negative_reviews = []
    aspect_negative_reviews = set(dictionary[aspect].Neg_reviews)
    r, p, accu = get_precision_recall_accuracy(aspect, aspect_positive_reviews, aspect_negative_reviews,
                                               true_aspect_and_it_reviews_dict)
    print(aspect, " recall= ", r, "   percision= ", p, "  accu= ", accu)
    aspects.append(aspect)
    recall.append(r)
    percison.append(p)
    accuracy.append(accu)
    f_measure.append(calculate_f_measure(r, p))

# Calculo de las métricas finales en promedio
rc, pr, f, accu = avg_acc_rec_per(recall, percison, f_measure, accuracy)
print("average recall= ", rc, "   average percision= ", pr, "     average f-measure= ", f, " aver accu= ", accu)
draw_graph(len(aspects), aspects, recall, percison, f_measure)
#pie_chart_graf(rc, pr, f)

# Generación del resumen de reseñas positivas y negativas por aspectos
for aspect in dictionary:
    print(aspect, 'positivas {}%'.format(round((len(dictionary[aspect].Pos_reviews) * 100) / (
            len(dictionary[aspect].Pos_reviews) + len(dictionary[aspect].Neg_reviews)), 2)), ' negativas {}%'.format(
        round((len(dictionary[aspect].Neg_reviews) * 100) / (
                len(dictionary[aspect].Pos_reviews) + len(dictionary[aspect].Neg_reviews)), 2)))
