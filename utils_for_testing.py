from utils_for_aspect_extraction import ClassifiedReviews
import numpy as np
import matplotlib.pyplot as plt


def read_all_the_clasified_reviews():
    """
    Este metodo lee todas las reseñas del archivo cámara, el cual posee la reseña, el aspecto y su clasificación de polaridad
    :return: un diccionario con key = aspecto aspect_positive_reviews aspect phrase y value = lista de instancias de la clase ClassifiedReviews()
    que tiene 2 atributos, la reseña y su polaridad
    """
    file = open("test.txt", "r")
    nouns_dic = {}
    phrases_dic = {}
    for line in file:
        if "##" in line:
            pos = line.index("##")
            rev = line[(pos + 2):]
            label = line[(pos - 3)]
            aspect = line[0:(pos - 4)]
            aspect = aspect.rstrip()
            rvew = ClassifiedReviews()
            rvew.review = rev
            rvew.pol = label
            if aspect.find(" ") != -1:
                if aspect not in phrases_dic.keys():
                    reviews = []
                    reviews.append(rvew)
                    phrases_dic[aspect] = reviews
                else:
                    if rvew not in phrases_dic[aspect]:
                        phrases_dic[aspect].append(rvew)
            else:
                if aspect not in nouns_dic.keys():
                    reviews = []
                    reviews.append(rvew)
                    nouns_dic[aspect] = reviews
                else:
                    if rvew not in nouns_dic[aspect]:
                        nouns_dic[aspect].append(rvew)

    for phrase in phrases_dic:
        splitted_phrase = phrase.split(" ")
        for nun in nouns_dic.keys():
            if nun in splitted_phrase:
                for aspect_review in nouns_dic[nun]:
                    if aspect_review not in phrases_dic[phrase]:
                        phrases_dic[phrase].append(aspect_review)
                for p in phrases_dic[phrase]:
                    if p not in nouns_dic[nun]:
                        nouns_dic[nun].append(p)

    dictionary = phrases_dic.copy()
    dictionary.update(nouns_dic)
    return dictionary


def avg_acc_rec_per(recall, percision, f_measure, accuracy):
    """
    Este método calcula el valor promedio de todas las métricas
    :param recall: lista que contiene el valor de recall para cada aspecto
    :param percision: lista que contiene el valor de precision para cada aspecto
    :param f_measure: lista que contiene el valor de f-measure para cada aspecto
    :param accuracy: lista que contiene el valor de accuracy para cada aspecto
    :return: average accuracy, recall, precision y f-measure
    """
    sum_rec = 0
    sum_per = 0
    sum_f = 0
    sum_accu = 0
    for i in accuracy:
        sum_accu += i
    avr_acc = sum_accu / len(accuracy)
    for i in recall:
        sum_rec += i
    avg_rec = sum_rec / len(recall)
    for i in percision:
        sum_per += i
    avg_per = sum_per / len(percision)
    for i in f_measure:
        sum_f += i
    avg_f = sum_f / len(f_measure)
    print("r= ", avg_rec, "p= ", avg_per, "f= ", avg_f)
    return round(avg_rec, 2) * 100, round(avg_per, 2) * 100, round(avg_f, 2) * 100, round(avr_acc, 3) * 100


def draw_graph(num, aspects, recall, percison, f_measure):
    """
    Esta función se utiliza para graficar los valores de precisión, recall y f-measure para cada uno de los
    aspectos extraídos
    :param num:
    :param aspects:
    :param recall:
    :param percison:
    :param f_measure:
    :return:
    """
    N = num
    ind = np.arange(N)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ass = np.asarray(aspects)

    rects2 = ax.bar(ind + width, recall, width, color='lightblue')
    rects3 = ax.bar(ind + width * 2, percison, width, color='steelblue')
    rects4 = ax.bar(ind + width * 3, f_measure, width, color='slategrey')
    ax.set_ylabel('Metrics scores')
    ax.set_xlabel('Aspects')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(ass)

    ax.legend((rects2[0], rects3[0], rects4[0]), ('Recall', 'Percision', 'F_measure'), loc='center left',
              bbox_to_anchor=(1, 0.5),
              fancybox=True, shadow=True, ncol=1)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.setp(ax.get_xticklabels(), rotation=35, horizontalalignment='right')
    plt.show()


def pie_chart_graf(avg_rc, avg_pr, aveg_f):
    """
    Esta función se utiliza para graficar las métricas en promedio general
    :param avg_rc:
    :param avg_pr:
    :param aveg_f:
    :return:
    """
    avg_recall = str(avg_rc) + "%"
    avg_percision = str(avg_pr) + "%"
    avegfmeasure = str(aveg_f) + "%"

    labels = [r'Recall  ' + '(' + str(avg_recall) + ')', r'Percision  ' + '(' + str(avg_percision) + ')',
              r'F-measure  ' + '(' + str(avegfmeasure) + ')']
    sizes = np.array([avg_rc, avg_pr, aveg_f])

    colors = ['lightblue', 'steelblue', 'slategrey']

    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()

    plt.show()


def get_precision_recall_accuracy(aspect, positives_lst, negatives_lst, dic):
    """
    Este método calcula las métricas para cada aspecto
    :param aspect: aspecto del cual se desea calcular la métrica
    :param positives_lst: lista de reseñas positivas extraídas de el aspecto
    :param negatives_lst: lista de reseñas negativas extraídas del aspecto
    :param dic: diccionario que contiene las reseñas y aspectos etiquetadas de forma manual
    :return: accuracy, recall, precision
    """
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    recall = 0
    precision = 0
    accuracy = 0
    polarities = []
    reviews = []

    if aspect in dic.keys():
        aspect_related_reviews = list(dic[aspect])
        for clasified_review in aspect_related_reviews:
            reviews.append(clasified_review.review.strip())
            polarities.append(clasified_review.pol)
        if len(positives_lst) > 0:
            for rvw in positives_lst:
                if (rvw.strip()) in reviews:
                    pol = polarities[reviews.index(rvw.strip())]
                    if pol == '+':
                        Tp += 1
                    else:
                        Fp += 1
        else:
            Tp = 0
            Fp = 0
        if len(negatives_lst) > 0:
            for rvw in negatives_lst:
                if rvw.strip() in reviews:
                    pol = polarities[reviews.index(rvw.strip())]
                    if pol == '-':
                        Tn += 1
                    else:
                        Fn += 1
        else:
            Tn = 0
            Fn = 0

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

    return recall, precision, accuracy


def calculate_f_measure(recall, precision):
    """
    Esta función calcula la medida F-1
    :param recall: exhaustividad obtenida del método get_acc
    :param precision: precision
    :return: f-measure
    """
    if (precision + recall) == 0:
        return 0
    else:
        return 2 * ((recall * precision) / (precision + recall))


def test_aspect_extraction(extracted_aspects, ignored_ones, labeled_reviews):
    Tp = 0
    Tn = 0
    Fp = 0
    Fn = 0
    precision = 0
    recall = 0
    accuracy = 0
    true_positives = []
    false_positives = []
    for aspect in extracted_aspects:
        if aspect in labeled_reviews:
            true_positives.append(aspect)
        else:
            false_positives.append(aspect)
    Tp = len(true_positives)
    Fp = len(false_positives)
    print('True positives')
    print(true_positives)
    print('False positives')
    print(false_positives)
    for aspect in ignored_ones:
        if aspect in labeled_reviews:
            Fn += 1
        else:
            Tn += 1
    if (Tp + Fp) == 0:
        precision = 0
    if (Tp + Fp) != 0:
        precision = Tp / (Tp + Fp)
        print('True positives', Tp)
        print('False positives', Fp)
    if (Tp + Fn) == 0:
        recall = 0
    if (Tp + Fn) != 0:
        recall = Tp / (Tp + Fn)
        print('False negatives', Fn)
    if (Tp + Fp + Tn + Fn) != 0:
        accuracy = (Tp + Tn) / (Tp + Fp + Tn + Fn)
        print('True negatives', Tn)

    return recall, precision, accuracy
