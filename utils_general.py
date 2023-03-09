import re
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from utils_for_aspect_extraction import TableAspectSynHypHypo, wrd_details
from utils_for_sentiment_clasification import get_semantic_similarity_adejectives, get_ratio212


def check_if_has_numbers(str):
    """
    Esta función comprueba si una palabra contiene números
    :param str: palabra que se desea comprobar
    :return: True si contiene numeros False si no
    """
    return bool(re.search(r'\d', str))


def read_unlabeled_reviews():
    """
    Esta función lee las resenas del archivo Canon G3
    :return: una lista con todas las reseñas sin etiquetar
    """
    reviews = []
    file = open("validate.txt", "r")

    for line in file:
        if "##" in line:
            pos = line.index("##")
            rev = line[(pos + 2):]
            reviews.append(rev)
    return reviews


def clean_the_review(noisy_text):
    """
    Esta función se utiliza en el preprocesamiento de las reseñas para eliminar caracteres innecesarios
    :param noisy_text: texto que se desea preprocesar
    :return: texto sin caracteres especiales ni numeros
    """
    out = "".join(c for c in noisy_text if c not in ('#', '$', '&', '!', '.', ':', '?', ';', '/', ',', '-', '(', ')'))
    string_no_numbers = re.sub("\d+", " ", out)
    return string_no_numbers


def remove_stop_words(review):
    """
    Esta función se utiliza para eliminar las stopwords y hacer la corrección ortográfica
    :param review: opinion de donde se desean eliminar las stopwords
    :return: reseña sin incongruencias
    """
    words = []
    stops = set(stopwords.words("english"))
    words2 = word_tokenize(review)
    for w in words2:
        words.append(w.lower())
    words_f = list(words)
    for w1 in words_f:
        for stp in stops:
            if (w1 == stp):
                words.remove(stp)
                break
    if "'ve" in words:
        words.remove("'ve")
    if "'ll" in words:
        words.remove("'ll")
    if "overall" in words:
        words.remove("overall")
    if "'re" in words:
        words.remove("'re")
    if "'d" in words:
        words.remove("'d")

    return words


def get_aspect_table_and_it_related_words(phrases, nouns):
    """
    Esta función construye diccionarios que contienen el aspecto (noun aspect_positive_reviews noun phrase) y para cada aspecto su hiperónimo,
    hipónimo y sinónimo
    :param phrases: diccionario key = noun phrases aspects values = noun phrases aspects structure
    :param nouns: lista de aspectos sustantivos
    :return: tres diccionarios que contienen el aspecto y sus palabras relacionadas
    """
    tbl_phrases_related_words = {}
    tbl_noun_and_related_words = {}
    tbl_phrases_in_detail = {}

    # Obtener el sinonimo, hiponimo e hiperonimo mas cercano para cada sustantivo
    for noun in nouns:
        aspect_related_words = TableAspectSynHypHypo()
        noun_synonym = wordnet.synsets(noun, wordnet.NOUN)
        nu_synsms = get_uppers_or_lowers_or_syno(noun, noun_synonym, 'syno')
        uppers = get_uppers_or_lowers_or_syno(noun, noun_synonym, 'hyper')
        lowers = get_uppers_or_lowers_or_syno(noun, noun_synonym, 'hypo')
        bst_lwr = get_best_word(noun, lowers)
        bst_upr = get_best_word(noun, uppers)
        bst_synm = get_best_word(noun, nu_synsms)
        aspect_related_words.synm = bst_synm
        aspect_related_words.upper = bst_upr
        aspect_related_words.lwr = bst_lwr
        tbl_noun_and_related_words[noun] = aspect_related_words

    for ph in phrases:
        aspect_related_words = TableAspectSynHypHypo()
        arr = ph.split()
        wrd1_syn = wordnet.synsets(arr[0], wordnet.NOUN)
        wrd2_syn = wordnet.synsets(arr[1], wordnet.NOUN)

        synsms_wrd2 = []
        synsms_wrd1 = []
        synsms_wrd1_ADJ = []
        lowers = []
        uppers = []
        lowers2 = []
        uppers2 = []
        synm_p1 = ""
        som1 = 0
        max1 = -1
        mx2 = 0
        sm2 = 0
        mx = 0
        sm = 0
        hypr1_adj = ""
        hyp1_adj = ""

        if (phrases[ph].prt1 == "JJ"):
            wrd1_syn_adj = wordnet.synsets(arr[0], pos='a')
            if (len(wrd1_syn_adj) != 0):
                for synset in wrd1_syn_adj:
                    for lemma in synset.lemmas():
                        l = lemma.name()
                        if (lemma.name() != arr[0] and lemma.name() not in synsms_wrd1_ADJ):
                            if ("_" in lemma.name()):
                                lema = lemma.name().replace("_", " ")
                                synsms_wrd1_ADJ.append(lema)

                            else:
                                synsms_wrd1_ADJ.append(lemma.name())
                for sn in wrd1_syn_adj:
                    hypo1 = sn.hyponyms()
                    hyper1 = sn.hypernyms()
                    for ho in hypo1:
                        for h in ho.lemmas():
                            str = ""

                            if ("_" in h.name()):
                                str = (h.name()).replace("_", " ")
                                if (str not in lowers):
                                    lowers.append(str)
                            else:
                                if (str not in lowers):
                                    lowers.append(h.name())
                    for hp in hyper1:
                        for hh in hp.lemmas():

                            str2 = ""
                            if ("_" in hh.name()):
                                str2 = (hh.name()).replace("_", " ")
                                if (str2 not in uppers):
                                    uppers.append(str2)

                            else:
                                if (str2 not in uppers):
                                    uppers.append(hh.name())
                for ss in synsms_wrd1_ADJ:
                    arr1 = ss.split()
                    if (len(arr1) == 1):
                        som1 = get_semantic_similarity_adejectives(arr[0], ss)
                    elif (len(arr1) == 2):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        som1 = b.wup_similarity(cb)
                    elif (len(arr1) == 3):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        som1 = b.wup_similarity(cb)
                    if (som1 != None):
                        if (som1 > max1):
                            max1 = som1
                            synm_p1 = ss
                    else:
                        continue
                for ss in lowers:
                    arr1 = ss.split()
                    if (len(arr1) == 1):
                        sm = get_semantic_similarity_adejectives(arr[0], ss)
                    elif (len(arr1) == 2):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        sm = b.wup_similarity(cb)
                    elif (len(arr1) == 3):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        sm = b.wup_similarity(cb)
                    if (sm != None):
                        if (sm > mx):
                            mx = sm
                            hyp1_adj = ss
                    else:
                        continue
                for ss in uppers:
                    arr1 = ss.split()
                    if (len(arr1) == 1):
                        sm2 = get_semantic_similarity_adejectives(arr[0], ss)
                    elif (len(arr1) == 2):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        sm2 = b.wup_similarity(cb)
                    elif (len(arr1) == 3):
                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".a.01")
                        b = wordnet.synset(arr[0] + ".a.01")
                        sm2 = b.wup_similarity(cb)
                    if (sm2 != None):
                        if (sm2 > mx2):
                            mx2 = sm2
                            hypr1_adj = ss
                    else:
                        continue
        elif (phrases[ph].prt1 != "JJ"):
            if (len(wrd1_syn) != 0):
                for synset in wrd1_syn:
                    for lemma in synset.lemmas():
                        if (lemma.name() != arr[0] and lemma.name() not in synsms_wrd1):
                            if ("_" in lemma.name()):
                                lema = lemma.name().replace("_", " ")
                                synsms_wrd1.append(lema)


                            else:
                                synsms_wrd1.append(lemma.name())
                for ss in synsms_wrd1:
                    arr1 = ss.split()
                    if (len(arr1) == 1):

                        som1 = get_ratio212(arr[0], ss)

                    elif (len(arr1) == 2):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        som1 = b.wup_similarity(cb)

                    elif (len(arr1) == 3):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        som1 = b.wup_similarity(cb)

                    if (som1 != None):
                        if (som1 > max1):
                            max1 = som1
                            synm_p1 = ss
                    else:
                        continue
                for sn in wrd1_syn:
                    hypo1 = sn.hyponyms()
                    hyper1 = sn.hypernyms()
                    for ho in hypo1:
                        for h in ho.lemmas():
                            str = ""
                            if ("_" in h.name()):
                                str = (h.name()).replace("_", " ")
                                if (str not in lowers):
                                    lowers.append(str)
                            else:
                                if (str not in lowers):
                                    lowers.append(h.name())
                    for hp in hyper1:
                        for hh in hp.lemmas():

                            str2 = ""
                            if ("_" in hh.name()):
                                str2 = (hh.name()).replace("_", " ")
                                if (str2 not in uppers):
                                    uppers.append(str2)

                            else:
                                if (str2 not in uppers):
                                    uppers.append(hh.name())
                for ss in lowers:
                    arr1 = ss.split()
                    if (len(arr1) == 1):

                        sm = get_semantic_similarity_adejectives(arr[0], ss)

                    elif (len(arr1) == 2):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        sm = b.wup_similarity(cb)

                    elif (len(arr1) == 3):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        sm = b.wup_similarity(cb)

                    if (sm != None):
                        if (sm > mx):
                            mx = sm
                            hyp1_adj = ss
                    else:
                        continue
                for ss in uppers:
                    arr1 = ss.split()
                    if (len(arr1) == 1):

                        sm2 = get_semantic_similarity_adejectives(arr[0], ss)

                    elif (len(arr1) == 2):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        sm2 = b.wup_similarity(cb)
                    elif (len(arr1) == 3):

                        cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                        b = wordnet.synset(arr[0] + ".n.01")
                        sm2 = b.wup_similarity(cb)

                    if (sm2 != None):
                        if (sm2 > mx2):
                            mx2 = sm2
                            hypr1_adj = ss
                    else:
                        continue
        if (len(wrd2_syn) != 0):
            synm_p2 = ""
            som2 = 0
            max2 = -1
            mxi2 = -1
            mxi22 = -1
            smi1 = 0
            smi2 = 0
            hypr2 = ""
            hyp2 = ""
            for synset in wrd2_syn:
                for lemma in synset.lemmas():
                    l = lemma.name()
                    if (lemma.name() != arr[1] and lemma.name() not in synsms_wrd2):
                        if ("_" in lemma.name()):
                            lema = lemma.name().replace("_", " ")
                            synsms_wrd2.append(lema)

                        else:
                            synsms_wrd2.append(lemma.name())
            for ss in synsms_wrd2:
                arr1 = ss.split()
                if (len(arr1) == 1):

                    som2 = get_ratio212(arr[1], ss)

                elif (len(arr1) == 2):

                    cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                    b = wordnet.synset(arr[1] + ".n.01")
                    som2 = b.wup_similarity(cb)

                elif (len(arr1) == 3):

                    cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                    b = wordnet.synset(arr[1] + ".n.01")
                    som2 = b.wup_similarity(cb)

                if (som2 != None):
                    if (som2 > max2):
                        max2 = som2
                        synm_p2 = ss
                else:
                    continue
            for sn in wrd2_syn:
                hypo2 = sn.hyponyms()
                hyper2 = sn.hypernyms()
                for ho in hypo2:
                    for h in ho.lemmas():
                        str = ""
                        if ("_" in h.name()):
                            str = (h.name()).replace("_", " ")
                            if (str not in lowers2):
                                lowers2.append(str)
                        else:
                            if (str not in lowers2):
                                lowers2.append(h.name())
                for hp in hyper2:
                    for hh in hp.lemmas():
                        str2 = ""
                        if ("_" in hh.name()):
                            str2 = (hh.name()).replace("_", " ")
                            if (str2 not in uppers2):
                                uppers2.append(str2)

                        else:
                            if (str2 not in uppers2):
                                uppers2.append(hh.name())
            for ss in lowers2:
                arr1 = ss.split()
                if (len(arr1) == 1):

                    smi1 = get_semantic_similarity_adejectives(arr[1], ss)

                elif (len(arr1) == 2):

                    cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                    b = wordnet.synset(arr[1] + ".n.01")
                    smi1 = b.wup_similarity(cb)

                elif (len(arr1) == 3):

                    cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                    b = wordnet.synset(arr[1] + ".n.01")
                    smi1 = b.wup_similarity(cb)

                if (smi1 != None):
                    if (smi1 > mxi2):
                        mxi2 = smi1
                        hyp2 = ss
                else:
                    continue
        for ss in uppers2:
            arr1 = ss.split()
            if (len(arr1) == 1):

                smi2 = get_semantic_similarity_adejectives(arr[1], ss)

            elif (len(arr1) == 2):

                cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + ".n.01")
                b = wordnet.synset(arr[1] + ".n.01")
                smi2 = b.wup_similarity(cb)

            elif (len(arr1) == 3):

                cb = wordnet.synset(arr1[0].strip() + "_" + arr1[1].strip() + "_" + arr1[2].strip() + ".n.01")
                b = wordnet.synset(arr[1] + ".n.01")
                smi2 = b.wup_similarity(cb)

            if (smi2 != None):
                if (smi2 > mxi22):
                    mxi22 = smi2
                    hypr2 = ss
            else:
                continue
        if (synm_p1 != "") and (synm_p2 != ""):
            bst_synom = synm_p1 + " " + synm_p2
        elif (synm_p1 == synm_p2):
            bst_synom = synm_p1
        elif (synm_p1 == ""):
            bst_synom = synm_p2
        elif (synm_p2 == ""):
            bst_synom = synm_p1
        if (hypr1_adj.strip() == hypr2.strip()):
            bst_upper = hypr1_adj
        elif (hypr1_adj != "") and (hypr2 != ""):
            bst_upper = hypr1_adj + " " + hypr2
        elif (hypr1_adj == ""):
            bst_upper = hypr2
        elif (hypr2 == ""):
            bst_upper = hypr1_adj
        if (hyp1_adj.strip() != hyp2.strip()):
            bst_lwr = hyp1_adj + " " + hyp2
        elif (hyp1_adj.strip() == hyp2.strip()):
            bst_lwr = hyp1_adj
        elif (hyp1_adj == ""):
            bst_lwr = hyp2
        elif (hyp2 == ""):
            bst_lwr = hyp1_adj

        wrd_det = wrd_details()
        if (synm_p1.strip() == synm_p2.strip()):
            wrd_det.synm_prt1 = synm_p1
            wrd_det.synm_prt2 = ""
        else:
            wrd_det.synm_prt1 = synm_p1
            wrd_det.synm_prt2 = synm_p2
        if (hypr2.strip() == hypr1_adj.strip()):
            wrd_det.hypr_prt1 = hypr1_adj
            wrd_det.hypr_prt2 = ""
        else:
            wrd_det.hypr_prt1 = hypr1_adj
            wrd_det.hypr_prt2 = hypr2
        if (hyp1_adj.strip() == hyp2.strip()):
            wrd_det.hypo_prt1 = hyp1_adj
            wrd_det.hypo_prt2 = ""
        else:
            wrd_det.hypo_prt1 = hyp1_adj
            wrd_det.hypo_prt2 = hyp2
        tbl_phrases_in_detail[ph] = wrd_det
        aspect_related_words.synm = bst_synom
        aspect_related_words.upper = bst_upper
        aspect_related_words.lwr = bst_lwr
        tbl_phrases_related_words[ph] = aspect_related_words
    return tbl_phrases_related_words, tbl_phrases_in_detail, tbl_noun_and_related_words


def get_best_word(noun_aspect, word_list):
    """
    Dado un aspecto y una lista de palabras la función retorna la palabra de la lista que mas se parece al aspecto dado
    :param noun_aspect: aspecto (noun)
    :param word_list: lista de palabras que se desea comparar con el aspecto
    :return: string palabra más parecida
    """
    best_lower = ''
    max_for_lower = -1
    smility_for_lower = 0
    for lower_word in word_list:
        splitted_lower_word = lower_word.split()

        if len(splitted_lower_word) == 1:
            smility_for_lower = get_semantic_similarity_adejectives(noun_aspect,
                                                                    lower_word)

        elif len(splitted_lower_word) == 2:

            noun_phrase_synsets = wordnet.synset(
                splitted_lower_word[0].strip() + "_" + splitted_lower_word[1].strip() + ".n.01")
            aspect_synset = wordnet.synset(noun_aspect + ".n.01")
            smility_for_lower = aspect_synset.wup_similarity(noun_phrase_synsets)

        elif len(splitted_lower_word) == 3:

            cb = wordnet.synset(splitted_lower_word[0].strip() + "_" + splitted_lower_word[1].strip() + "_" +
                                splitted_lower_word[2].strip() + ".n.01")
            b = wordnet.synset(noun_aspect + ".n.01")
            smility_for_lower = b.wup_similarity(cb)

        if smility_for_lower is not None:
            if smility_for_lower > max_for_lower:
                max_for_lower = smility_for_lower
                best_lower = lower_word
        else:
            continue
    return best_lower


def get_uppers_or_lowers_or_syno(noun_aspect, synsets, kind):
    lowers = []
    uppers = []
    synonyms = []
    if len(synsets) > 0:
        if kind == 'syno':
            for nu in synsets:  # En esta parte se guardan todos los sinónimos en una lista
                for lemma in nu.lemmas():
                    if lemma.name() != noun_aspect and lemma.name() not in synonyms:
                        if "_" in lemma.name():
                            lema = lemma.name().replace("_", " ")
                            synonyms.append(lema)
                        else:
                            synonyms.append(lemma.name())
            return synonyms

        if kind == 'hypo':
            for synset in synsets:
                hypos = synset.hyponyms()
                for hyponym in hypos:
                    for hyponym_lemma in hyponym.lemmas():
                        hyponym_name = ""
                        if "_" in hyponym_lemma.name():
                            hyponym_name = (hyponym_lemma.name()).replace("_", " ")
                            if hyponym_name not in lowers:
                                lowers.append(hyponym_name)
                        else:
                            if hyponym_name not in lowers:
                                lowers.append(hyponym_lemma.name())

                return lowers
        if kind == 'hyper':
            for synset in synsets:
                hypers = synset.hypernyms()
                for hypernym in hypers:
                    for hypernym_lemma in hypernym.lemmas():
                        hypernym_name = ""
                        if "_" in hypernym_lemma.name():
                            hypernym_name = (hypernym_lemma.name()).replace("_", " ")
                            if hypernym_name not in uppers:
                                uppers.append(hypernym_name)
                        else:
                            if hypernym_name not in uppers:
                                uppers.append(hypernym_lemma.name())
                return uppers
