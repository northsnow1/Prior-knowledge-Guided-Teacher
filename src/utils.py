"""
@Time   :   2021-01-12 15:10:43
@File   :   utils.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import json
import os
import sys, random
import numpy as np


def compute_corrector_prf(results):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_predict_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1

        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)
        all_predict_index.append(predict_index)
    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                             detection_f1))

    TP_C = 0
    FP_C = 0
    FN_C = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP_C += 1
                else:
                    FP_C += 1
        for j in all_gold_index[i]:
            # if results[i][1][j] in predict_words:
            if results[i][1][j] == results[i][2][j]:
                continue
            else:
                FN_C += 1
    print(TP_C, FP_C, FN_C)
    # For the correction Precision, Recall and F1
    correction_precision = TP_C / (TP_C + FP_C) if (TP_C + FP_C) > 0 else 0
    correction_recall = TP_C / (TP_C + FN_C) if (TP_C + FN_C) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    print("The correction result is precision={}, recall={} and F1={}".format(correction_precision,
                                                                              correction_recall,
                                                                              correction_f1))

    return detection_f1, correction_f1


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(obj, fp):
    try:
        fp = os.path.abspath(fp)
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        print(f'json文件保存成功，{fp}')
        return True
    except Exception as e:
        print(f'json文件{obj}保存失败, {e}')
        return False


def get_main_dir():
    # 如果是使用pyinstaller打包后的执行文件，则定位到执行文件所在目录
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    # 其他情况则定位至项目根目录
    return os.path.join(os.path.dirname(__file__), '..')


def get_abs_path(*name):
    # print(os.path.abspath(os.path.join(get_main_dir(), *name)))
    return os.path.abspath(os.path.join(get_main_dir(), *name))

import errant
def compute_sentence_level_prf(results, tokenizer):
    """
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    corrected_sent = 0.0
    correct_sent = 0.0
    TP_d = 0.0
    FP_d = 0.0
    FN_d = 0.0
    TN_d = 0.0
    i=0
    total_num = len(results)

    for item in results:
        src, tgt, predict = item
        l = len(list(src))
        gold_index = []
        for i in range(l):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        # all_gold_index.append(gold_index)
        predict_index = []
        for i in range(l):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        src = tokenizer.decode(src).replace(' ', '')
        predict = tokenizer.decode(predict).replace(' ', '')
        tgt = tokenizer.decode(tgt).replace(' ', '')
        if src == tgt:
            if gold_index == predict_index:
                TN_d += 1
            else:
                FP_d += 1
        else:
            if gold_index == predict_index:
                TP_d += 1
            else:
                FN_d += 1

        if src == tgt:
            if src == predict:
                TN += 1
            else:
                FP += 1

        else:
            if tgt == predict:
                with open('./tp.txt', 'a+', encoding='utf-8') as f:
                    f.write(src)
                    f.write('\t')
                    f.write(tgt)
                    f.write('\t')
                    f.write(predict)
                    f.write('\n')
                TP += 1
            else:
                FN += 1
                i += 1

    acc_d = (TP_d + TN_d) / total_num
    fpr_d = FP_d / (FP_d + TN_d) if ((FP_d + TN_d)) != 0 else 0
    precision_d = TP_d / (TP_d + FP_d) if (TP_d + FP_d) != 0 else 0
    recall_d = TP_d / (TP_d + FN_d) if (TP_d + FN_d) != 0 else 0
    f1_d = 2 * precision_d * recall_d / (precision_d + recall_d) if precision_d + recall_d != 0 else 0
    # print(i)
    print(f'Sentence Level:fpr_d:{fpr_d:.6f}, acc_d:{acc_d:.6f}, precision_d:{precision_d:.6f}, recall_d:{recall_d:.6f}, f1_d:{f1_d:.6f}')


    acc = (TP + TN) / total_num
    fpr = FP / (FP + TN) if ((FP + TN)) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    print(TP, FN, FP, TN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    # print(i)
    print(f'Sentence Level:fpr:{fpr:.6f}, acc:{acc:.6f}, precision:{precision:.6f}, recall:{recall:.6f}, f1:{f1:.6f}')
    return acc, precision, recall, f1


def compute_det_prf(results):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    TP1 = 0
    FN1 = 0
    TP2 = 0
    FN2 = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, label, predict = item
        gold_index = []
        label1 = label[1:len(src) + 1].cpu().numpy().tolist()
        each_true_index = []
        if np.sum(label1) == 0:
            if predict == label1:
                TN += 1
            else:
                FP += 1
        elif np.sum(label1) == 1:
            if predict == label1:
                TP1 += 1
            else:
                FN1 += 1
        elif np.sum(label1) > 1:
            if predict == label1:
                TP2 += 1
            else:
                FN2 += 1

    TP = TP1 + TP2
    FN = FN1 + FN2
    # For the detection Precision, Recall and F1
    print(TP, TN, FP, FN, FN1, FN2)
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                             detection_f1))


import sys, os
import numpy as np


os.environ["PYTHONIOENCODING"] = "utf-8"


def score_f(results, print_flg=False, only_check=False, out_dir=''):
    fout = open('%s/pred.txt' % out_dir, 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    # inputs, golds, preds = ans
    for item in results:
        ori, god, prd = item
        for i in range(len(ori)):
            if ori[i] != god[i]:
                total_gold_err += 1
            if prd[i] != ori[i]:
                total_pred_err += 1
            if (ori[i] != god[i]) and (prd[i] != ori[i]):
                check_right_pred_err += 1
                if god[i] == prd[i]:
                    right_pred_err += 1

    p = 1. * check_right_pred_err / (total_pred_err + 0.0001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.0001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token num: gold_n:%d, pred_n:%d, right_n:%d' % (total_gold_err, total_pred_err, check_right_pred_err))
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    pc1 = 1. * right_pred_err / (total_pred_err + 0.0001)
    pc2 = 1. * right_pred_err / (check_right_pred_err + 0.0001)
    rc = 1. * right_pred_err / (total_gold_err + 0.0001)
    fc1 = 2 * pc1 * rc / (pc1 + rc + 1e-13)
    fc2 = 2 * pc2 * rc / (pc2 + rc + 1e-13)
    print('token correction-1: p=%.3f, r=%.3f, f=%.3f' % (pc2, rc, fc2))
    print('token correction-2: p=%.3f, r=%.3f, f=%.3f' % (pc1, rc, fc1))
    return p, r, f

def score_f_sent(results):

    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    # fout = open('sent_pred_result.txt', 'w', encoding='utf-8')
    for item in results:
        ori_tags, god_tags, prd_tags = item
        gold_errs = [idx for (idx, tk) in enumerate(god_tags) if tk != ori_tags[idx]]
        pred_errs = [idx for (idx, tk) in enumerate(prd_tags) if tk != ori_tags[idx]]

        if len(gold_errs) > 0:
            total_gold_err += 1
            # fout.writelines('gold_err\n')
        if len(pred_errs) > 0:
            # fout.writelines('check_err\n')
            total_pred_err += 1
            if gold_errs == pred_errs:
                check_right_pred_err += 1
                # fout.writelines('check_right\n')
            if god_tags == prd_tags:
                right_pred_err += 1
                # fout.writelines('correct_right\n')
    # fout.close()
    p = 1. * check_right_pred_err / total_pred_err
    r = 1. * check_right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    # print(total_gold_err, total_pred_err, right_pred_err, check_right_pred_err)
    print('sent check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    p = 1. * right_pred_err / total_pred_err
    r = 1. * right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    return p, r, f