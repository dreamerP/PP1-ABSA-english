import inflection
import nltk
from utils_for_aspect_extraction import AspectPhraseStructure, tf, get_tag
from textblob import TextBlob as tb
from tqdm import tqdm

def get_most_frequent_aspects(reviews):
    """
    Esta función extrae los sustantivos y las frases sustantivas que más frecuentemente se utilicen en el texto
    :param reviews: reseñas de los usuarios :return: lista de sustantivos y lista de frases sustantivas más usadas y
    diccionario que contiene todas las frases sustantivas con su estructura
    """
    aspect_phrase_plus_strucure = {}
    reviews_aspects = []
    review_phrases = []
    all_rev = ""

    grammar_aspects = """NP: {<NN><NN>}
                             {<NNS><NNS>}
                             {<NN><NNS>}
                             {<NNS><NN>}
                             {<JJ><NN>}
                             {<JJ><NNS>}"""

    chunk_parser = nltk.RegexpParser(grammar_aspects)
    for review in reviews:
        all_rev += review + " "
        tokenized = nltk.word_tokenize(review)
        tagged_words = nltk.pos_tag(tokenized)
        preprocessed_review = chunk_parser.parse(tagged_words)
        for word in preprocessed_review:
            if "NN" in word or "NNS" in word:
                if ":)" not in word:
                    if word[0] not in reviews_aspects and len(word[0]) > 1:
                        reviews_aspects.append(word[0])

            if str(word).find("NP") != -1:
                aspect_phrase_structure = AspectPhraseStructure()
                word1 = str(word[0]).replace("('", "")
                word1 = word1.replace("',", "")
                if word1.find("'JJ')") != -1:
                    word1 = word1.replace("'JJ')", "")
                    part1 = 'JJ'
                if word1.find("'NN')") != -1:
                    word1 = word1.replace("'NN')", "")
                    part1 = 'NN'
                word2 = str(word[1]).replace("('", "")
                word2 = word2.replace("',", "")
                if word2.find("'NN')") != -1:
                    word2 = word2.replace("'NN')", "")
                    part2 = 'NN'
                if word2.find("'NNS')") != -1:
                    word2 = word2.replace("'NNS')", "")
                    part2 = 'NNS'
                noun_phrase = str(word1.strip()) + " " + str(word2.strip())
                aspect_phrase_structure.prt1 = part1
                aspect_phrase_structure.prt2 = part2
                aspect_phrase_plus_strucure[noun_phrase] = aspect_phrase_structure
                if noun_phrase not in review_phrases and len(word1.strip()) > 1 and len(word2.strip()) > 1:
                    review_phrases.append(noun_phrase)

    blob = tb(all_rev)
    noun_aspects = []
    phrase_aspects = []
    in_singular_phrase_aspects = []
    not_related = []
    print('Todos los sustantivos identificados fueron:')
    print(reviews_aspects)
    print('Todos las frases sustantivas identificados fueron:')
    print(review_phrases)
    for noun in reviews_aspects:  # Por cada aspecto que exista en la lista de sustantivos extraidos
        freq = tf(noun, blob)  # Contar la frecuencia del aspecto
        singular = inflection.singularize(noun)  # Llevar el sustantivo a singular
        if freq > 15 and get_tag(
                singular) == 'NN':  # Si la frecuncia de aparición es mayor que 24 y el susutantivo en singular sigue siendo un sustantivo
            if singular not in noun_aspects:  # Si todavía no se encuentra en la lista de aspectos se añade
                noun_aspects.append(singular)
        elif freq <= 15 and noun not in not_related:
            not_related.append(noun)

    noun_aspects = set(noun_aspects)
    for aspect_phrase in tqdm(review_phrases, desc = 'Calculando frecuencia de aparición de las frases sustantivas: '):
        counter = 0
        for review in reviews:
            if review.find(aspect_phrase) != -1 or review.find(
                    inflection.pluralize(aspect_phrase)) != -1 or review.find(
                inflection.singularize(aspect_phrase)) != -1:
                counter += 1
        if counter > 4:
            if aspect_phrase not in phrase_aspects:
                phrase_aspects.append(aspect_phrase)
        elif counter <= 4 and aspect_phrase not in not_related:
            not_related.append(aspect_phrase)

    for phrase in phrase_aspects:
        in_singular_phrase_aspects.append(
            inflection.singularize(phrase))  # Se crea una lista con los noun phrases en singular
    in_singular_phrase_aspects = set(in_singular_phrase_aspects)
    nouns_final = list(noun_aspects)
    print('Las frases sustantivas más frecuentes son: ')
    for in_singular_form in in_singular_phrase_aspects:
        words = in_singular_form.split()
        word_1_of_phrase = words[0]
        word_2_of_phrase = words[1]
        print(words)
        try:
            if word_1_of_phrase in nouns_final and word_2_of_phrase in nouns_final:
                nouns_final.remove(word_1_of_phrase)
                nouns_final.remove(word_2_of_phrase)
        except:
            if word_1_of_phrase not in nouns_final and word_2_of_phrase not in nouns_final:
                print
                print("NO")
    in_singular_form = set(in_singular_phrase_aspects)
    for singular in in_singular_form:
        phrase = singular.split()
        word1 = phrase[0]
        word2 = phrase[1]
        if get_tag(word1) == 'JJ':
            in_singular_phrase_aspects.remove(singular)
            if (word2 not in nouns_final) and (tf(word2, blob) > 15):
                nouns_final.append(word2)
    phrases_finals = []
    for word1 in in_singular_phrase_aspects:
        phrases_finals.append(word1.rstrip())

    return nouns_final, phrases_finals, aspect_phrase_plus_strucure, not_related
