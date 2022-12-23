import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import auc


def is_in(triple1, lst, k=3):
    """ Performs a soft matching to get rid of tokenization artifacts. """
    triple1 = '{}#{}#{}#{}'.format(*triple1)
    for triple2 in lst:
        triple2 = '{}#{}#{}#{}'.format(*triple2)
        if levenshtein_distance(triple1, triple2) <= k:
            return True
    return False


def precision_at_k(y_true, y_pred, k):
    tp, fp, = 0, 0
    for triples_true, triples_pred in zip(y_true, y_pred):
        # Filter out triples below k
        triples_pred = set([triple for conf, triple in triples_pred if conf >= k])

        for triple in triples_pred:
            if is_in(triple, triples_true):
                tp += 1
            else:
                fp += 1

    # Catch divide by zero
    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall_at_k(y_true, y_pred, k):
    tp, fn, = 0, 0
    for triples_true, triples_pred in zip(y_true, y_pred):
        # Filter out triples below k
        triples_pred = set([triple for conf, triple in triples_pred if conf >= k])

        for triple in triples_true:
            if is_in(triple, triples_pred):
                tp += 1
            else:
                fn += 1

    # Catch divide by zero
    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def f_score_at_k(y_true, y_pred, k):
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def precision_recall_auc(y_true, y_pred, steps=2000, plot_pr=True):
    # Get list of confidences
    flat_triples = [t for ts in y_pred for t in ts]
    confidences = sorted(list(set([c for c, _ in flat_triples])))

    print(confidences)
    precision, recall = [], []
    for k in confidences:
        precision.append(precision_at_k(y_true, y_pred, k))
        recall.append(recall_at_k(y_true, y_pred, k))

    # Ensure complete curve
    precision.append(1)
    recall.append(0)

    if plot_pr:
        plt.grid('on', c='lightgrey')
        plt.plot([0, 1], [1, 0], 'r--', linewidth=1)
        plt.plot(recall, precision)
        plt.xlabel('recall (R)')
        plt.ylabel('precision (P)')
        plt.show()

    return auc(recall, precision), (recall, precision)


def classification_report(true_triples, pred_triples, k):
    """ Computes precision@k, recall@k, F1@k and PR-AUC over gold
        triples and predictions.

    :param true_triples:
    :param pred_triples:
    :param k:
    :return:
    """
    precision = precision_at_k(true_triples, pred_triples, k)
    recall = recall_at_k(true_triples, pred_triples, k)
    fscore = f_score_at_k(true_triples, pred_triples, k)
    auc, pr_curve = precision_recall_auc(true_triples, pred_triples)

    print('precision@k:', precision)
    print('recall@k:   ', recall)
    print('F-score@k:  ', fscore)
    print('AUC:        ', auc)
    return precision, recall, fscore, auc, pr_curve


if __name__ == '__main__':
    """ Toy example sentences
        sent1: I like cats
        sent2: I despise aliens
        sent3: I went for a walk
        sent4: I like skittles
        sent5: Candy are not good and bad
        sent6: You are a hyper like my cats
    """
    true_triples = [[("I", "like", "cats")],
                    [("I", "despise", "aliens")],
                    [("I", "went for", "a walk")],
                    [("I", "like", "skittles"), ("I", "like", "candy")]]

    pred_triples = [[(0.1, ("I", "like", "cats"))],
                    [(0.2, ("I", "despise", "aliens"))],
                    [(0.4, ("I", "went for", "a walk")), (0.5, ("I", "went for", "many walks"))],
                    [(0.8, ("I", "like", "skittles"))]]

    print('precision@k:', precision_at_k(true_triples, pred_triples, k=0.7))
    print('recall@k:   ', recall_at_k(true_triples, pred_triples, k=0.7))
    print('F-score@k:  ', f_score_at_k(true_triples, pred_triples, k=0.7))
    print('AUC:        ', precision_recall_auc(true_triples, pred_triples))