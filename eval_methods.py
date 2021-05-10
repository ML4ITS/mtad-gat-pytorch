import numpy as np
import more_itertools as mit
from spot import SPOT, dSPOT


def adjust_predicts(score, label, threshold, advance=1, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    # Added advance in case model predicts anomaly 'in advance' within a small window
    # Advance should be 0 or small
    for i in range(len(predict)):
        if any(actual[max(i - advance, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, q=1e-3, level=0.99):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=False)  # initialization step
    ret = s.run(dynamic=False, with_alarm=False)  # much faster

    # s = dSPOT(q, depth=300)  # SPOT object
    # s.fit(init_score, score)  # data import
    # s.initialize()  # initialization step
    # ret = s.run()  # much faster

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            'threshold': ret["thresholds"],
        }


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
            list: list for results
            float: the `threshold` for best-f1
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, latency = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "latency": m_l,
    }


def calc_seq(score, label, threshold):
    """
    Calculate f1 score for a score sequence
    """
    predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    return calc_point2point(predict, label), latency


def epsilon_eval(train_score, test_score, label):
    val_split = 0.2 if label is not None else 0.0
    val_end = int(val_split * len(test_score))
    val_score = test_score[:val_end]
    val_label = label[:val_end]
    test_score = test_score[val_end:]
    test_label = label[val_end:]
    best_reg = 0

    if label is None:
        best_epsilon = find_epsilon(train_score)
    else:
        reg_levels = [0, 1, 2]
        best_epsilon = None
        best_f1 = -1
        best_acc = -1
        for reg in reg_levels:
            epsilon = find_epsilon(train_score, reg_level=reg)
            val_pred = adjust_predicts(val_score, val_label, epsilon, calc_latency=False)
            p_t = calc_point2point(val_pred, val_label)
            acc = ( p_t[3] + p_t[4]) / (p_t[3] + p_t[4] + p_t[5] + p_t[6])
            f1 = p_t[0]
            if 1 in val_label:
                if f1 > best_f1:
                    best_f1 = f1
                    best_epsilon = epsilon
                    best_reg = reg
            else:
                if acc > best_acc:
                    best_acc = acc
                    best_epsilon = epsilon
                    best_reg = reg

    pred, p_latency = adjust_predicts(test_score, test_label, best_epsilon, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, test_label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "latency": p_latency,
            'reg_level': best_reg
        }
    else:
        return {
            'threshold': best_epsilon,
            'reg_level': best_reg
        }

def find_epsilon(errors, reg_level=0):
    e_s = errors

    sd_threshold = None
    best_epsilon = None

    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1, )
        buffer = np.arange(1, 50)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)**2
            elif reg_level == 2:
                denom = (len(E_seq) ** 2 + len(i_anom)**2)

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            # sanity checks / guardrails
            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                sd_threshold = z
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon