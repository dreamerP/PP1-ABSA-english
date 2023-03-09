import re
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.corpus import wordnet_ic
import numpy as np
from sklearn.preprocessing import normalize


class wrd_details:
    def __init__(self):
        self.synm_prt1 = ""
        self.synm_prt2 = ""
        self.hypr_prt1 = ""
        self.hypr_prt2 = ""
        self.hypo_prt1 = ""
        self.hypo_prt2 = ""


class AspectPhraseStructure:
    def __init__(self):
        self.prt1 = ""
        self.prt2 = ""


class ClassifiedReviews:
    def __init__(self):
        self.review = ""
        self.pol = ""


class TableAspectSynHypHypo:
    def __init__(self):
        self.upper = ""
        self.synm = ""
        self.lwr = ""


def get_tagy(opinion_word, sentence):
    """
    Esta función recibe como parámetro un término de opinión una oración y retorna el POS tag del término
    :param opinion_word: término que se utiliza para emitir la opinión
    :param sentence: oración donde se encuentra la opinión
    :return: POS tag
    """
    splitted_sentence = sentence.split()
    words2 = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words2)
    ind = -1
    if opinion_word in splitted_sentence:
        ind = splitted_sentence.index(opinion_word)
    if ind != -1:
        return tagged_words[ind][1]
    else:
        return "error"


def get_tag(opinion_word):
    """
    Esta función retorna la POS tag de una palabra
    :param opinion_word: palabra de la cual se desea conocer el POS tag
    :return: POS tag de una determinada palabra
    """
    lst = nltk.tag.pos_tag([opinion_word])
    return lst[0][1]


def tf(word, blob):
    """
    Esta función permite obtener la frecuencia de una palabra determinada
    :param word: palabra de la que se quiere conocer la frecuencia
    :param blob: texto construido con text blob
    :return: frecuencia de aparición de la palabra en el texto
    """
    return blob.words.count(word)


def zimmerman(normalized_vector):
    T_en = 1
    T_e_complement = 1
    for m in normalized_vector:
        if m > 0:
            T_en *= m
            T_e_complement *= 1 - m

    gamma = T_en / (T_en + T_e_complement)
    mul1 = T_en ** (1 - gamma)
    mul2 = 1 - (T_e_complement ** gamma)
    result = mul1 * mul2
    return result


def get_normalized_vector(word):
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    domain_word = 'camera'
    word_to_compare = word.lower()
    domain_word_synset = wordnet.synset(str(domain_word) + ".n.01")
    try:
        word_to_compare_synset = wordnet.synset(word_to_compare.strip() + ".n.01")
    except:
        pass
    try:
        wu_palmer_sim = domain_word_synset.wup_similarity(word_to_compare_synset)
    except:
        wu_palmer_sim = 0
    try:
        j_and_c = wordnet.jcn_similarity(domain_word_synset, word_to_compare_synset, ic=brown_ic)
    except:
        j_and_c = 0
    try:
        resnik = wordnet.res_similarity(domain_word_synset, word_to_compare_synset, ic=brown_ic)
    except:
        resnik = 0
    try:
        lin = wordnet.lin_similarity(domain_word_synset, word_to_compare_synset, ic=brown_ic)
    except:
        lin = 0

    metrics = np.array([lin, resnik, wu_palmer_sim, j_and_c])
    normalized_vecto = normalize(metrics[:, np.newaxis], axis=0)
    normalized_vector = []
    for n in normalized_vecto:
        normalized_vector.append(n[0])
    return normalized_vector


def preprocessor(reiew):
    clean_text = reiew.lower()  # Transform to lowercase
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\-\\.\\/\\:\\;\\<\\,\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    clean_text = re.sub(regex, ' ', clean_text)  # Erase the punctuation
    clean_text = re.sub("\d+", ' ', clean_text)  # Erase numbers
    clean_text = re.sub("\\s+", ' ', clean_text)  # Erase multiple whitespace
    clean_text = word_tokenize(clean_text, 'english')
    all_stopwords = stopwords.words('english')
    clean_text = [word for word in clean_text if word not in all_stopwords]
    return clean_text
