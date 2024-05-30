"""
@Time   :   2021-01-12 15:23:38
@File   :   data_processor.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import gc
import random
from lxml import etree

import opencc
from tqdm import tqdm
import os
from .utils import dump_json, get_abs_path


def proc_item(item, convertor):
    root = etree.XML(item)
    passages = dict()
    mistakes = []
    for passage in root.xpath('/ESSAY/TEXT/PASSAGE'):
        passages[passage.get('id')] = convertor.convert(passage.text)
    for mistake in root.xpath('/ESSAY/MISTAKE'):
        mistakes.append({'id': mistake.get('id'),
                         'location': int(mistake.get('location')) - 1,
                         'wrong': convertor.convert(mistake.xpath('./WRON G/text()')[0].strip()),
                         'correction': convertor.convert(mistake.xpath('./CORRECTION/text()')[0].strip())})

    rst_items = dict()
    for mistake in mistakes:
        if mistake['id'] not in rst_items.keys():
            rst_items[mistake['id']] = {'original_text': passages[mistake['id']],
                                        'wrong_ids': [],
                                        'correct_text': passages[mistake['id']]}

        # todo 繁体转简体字符数量或位置发生改变校验

        ori_text = rst_items[mistake['id']]['original_text']
        cor_text = rst_items[mistake['id']]['correct_text']
        if len(ori_text) == len(cor_text):
            if ori_text[mistake['location']] in mistake['wrong']:
                rst_items[mistake['id']]['wrong_ids'].append(mistake['location'])
                wrong_char_idx = mistake['wrong'].index(ori_text[mistake['location']])
                start = mistake['location'] - wrong_char_idx
                end = start + len(mistake['wrong'])
                rst_items[mistake['id']][
                    'correct_text'] = f'{cor_text[:start]}{mistake["correction"]}{cor_text[end:]}'
        else:
            print(f'{mistake["id"]}\n{ori_text}\n{cor_text}')
    rst = []
    for k in rst_items.keys():
        if len(rst_items[k]['correct_text']) == len(rst_items[k]['original_text']):
            rst.append({'id': k, **rst_items[k]})
        else:
            text = rst_items[k]['correct_text']
            rst.append({'id': k, 'correct_text': text, 'original_text': text, 'wrong_ids': []})
    return rst


def proc_test_set(fp, convertor):
    """
    生成sighan15的测试集
    Args:
        fp:
        convertor:

    Returns:

    """
    inputs = dict()
    with open(os.path.join(fp, 'test15_gcn.txt'), 'r', encoding='utf8') as f:
        for line in f:
            pid = line.split()[0][5:-1]
            text = line.split()[1].strip()
            inputs[pid] = text

    rst = []
    rst_mlt = []
    with open(os.path.join(fp, 'testtruth15_gcn.txt'), 'r', encoding='utf8') as f:
        for line in f:
            split_line = line.strip().split(', ')
            pid = split_line[0]
            mistakes = split_line[1:]
            if len(mistakes) <= 1:
                text = convertor.convert(inputs[pid])
                rst.append({'id': pid,
                            'original_text': text,
                            'wrong_ids': [],
                            'correct_text': text})
            else:
                wrong_ids = []
                original_text = convertor.convert(inputs[pid])
                cor_text = inputs[pid]
                for i in range(len(mistakes) // 2):
                    idx = int(mistakes[2 * i]) - 1
                    cor_char = mistakes[2 * i + 1]
                    wrong_ids.append(idx)
                    cor_text = f'{cor_text[:idx]}{cor_char}{cor_text[idx + 1:]}'
                # original_text = convertor.convert(original_text)
                # cor_text = convertor.convert(cor_text)
                if len(original_text) != len(cor_text):
                    print(pid)
                    print(original_text)
                    print(cor_text)
                    continue


                rst.append({
                    'id': '-',
                    'original_text': original_text,
                    'wrong_ids': wrong_ids,
                    'correct_text': cor_text
                })

    return rst


def read_data(fp):
    for fn in os.listdir(fp):
        if fn.endswith('ing.sgml'):
            print(fn)
            with open(os.path.join(fp, fn), 'r', encoding='utf-8') as f:
                item = []
                for line in f:
                    if line.strip().startswith('<ESSAY') and len(item) > 0:
                        yield ''.join(item)
                        item = [line.strip()]
                    elif line.strip().startswith('<'):
                        item.append(line.strip())

def read_data1(fp):
    for fn in os.listdir(fp):
        if fn.endswith('ng.sgml'):
            print(fn)
            with open(os.path.join(fp, fn), 'r', encoding='utf-8') as f:
                item = []
                for line in f:
                    if line.strip().startswith('<SENT') and len(item) > 0:
                        yield ''.join(item)
                        item = [line.strip()]
                    elif line.strip().startswith('<'):
                        item.append(line.strip())



def read_confusion_data(fp):
    fn = os.path.join(fp, 'train.sgml')
    # print('hh'+fn)
    with open(fn, 'r', encoding='utf-8') as f:
        item = []
        for line in tqdm(f):
            if line.strip().startswith('<SENT') and len(item) > 0:
                yield ''.join(item)
                item = [line.strip()]

            elif line.strip().startswith('<'):
                item.append(line.strip())

def read_test13_data(fp):
    fn = os.path.join(fp, 'test13.sgml')
    # print('hh'+fn)
    with open(fn, 'r', encoding='utf-8') as f:
        item = []
        i=0
        for line in tqdm(f):
            # print(line)
            if line.strip().startswith('<SENT') and len(item) > 0:
                # print(line)
                yield ''.join(item)
                # i+=1
                # print(i)
                item = [line.strip()]
            elif line.strip().startswith('<'):
                item.append(line.strip())

def read_test14_data(fp):
    fn = os.path.join(fp, 'test14.sgml')
    # print('hh'+fn)
    with open(fn, 'r', encoding='utf-8') as f:
        item = []
        for line in tqdm(f):
            if line.strip().startswith('<SENT') and len(item) > 0:
                yield ''.join(item)
                item = [line.strip()]
            elif line.strip().startswith('<'):
                item.append(line.strip())

def read_test15_data(fp):
    fn = os.path.join(fp, 'test15.sgml')
    # print('hh'+fn)
    with open(fn, 'r', encoding='utf-8') as f:
        item = []
        for line in tqdm(f):
            if line.strip().startswith('<SENT') and len(item) > 0:
                yield ''.join(item)
                item = [line.strip()]
            elif line.strip().startswith('<'):
                item.append(line.strip())

def proc_confusion_item(item):
    """
    处理confusionset数据集
    Args:
        item:

    Returns:

    """
    root = etree.XML(item)
    text = root.xpath('/SENTENCE/TEXT/text()')[0]
    mistakes = []
    for mistake in root.xpath('/SENTENCE/MISTAKE'):
        mistakes.append({'location': int(mistake.xpath('./LOCATION/text()')[0]) - 1,
                         'wrong': mistake.xpath('./WRONG/text()')[0].strip(),
                         'correction': mistake.xpath('./CORRECTION/text()')[0].strip()})

    cor_text = text
    wrong_ids = []
    rst = []
    for mis in mistakes:
        cor_text = f'{cor_text[:mis["location"]]}{mis["correction"]}{cor_text[mis["location"] + 1:]}'
        wrong_ids.append(mis['location'])
    if len(wrong_ids) != 0:
        if random.random() < 0:
            rst = [{'id': '--',
                    'original_text': cor_text,
                    'wrong_ids': [],
                    'correct_text': cor_text}]
        else:
            rst = [{
                'id': '-',
                'original_text': text,
                'wrong_ids': wrong_ids,
                'correct_text': cor_text
            }]
    else:
        rst = [{
            'id': '-',
            'original_text': text,
            'wrong_ids': wrong_ids,
            'correct_text': cor_text
        }]
    if len(text) != len(cor_text):
        print(text)
        print(cor_text)
        return [{'id': '--',
                 'original_text': cor_text,
                 'wrong_ids': [],
                 'correct_text': cor_text}]
        
    return rst


def preproc():
    rst_items = []
    convertor = opencc.OpenCC('tw2sp.json')
    test_items = proc_test_set('data', convertor)
    for item in read_data(get_abs_path('data')):
        rst_items += proc_item(item, convertor)
    print(len(rst_items))
    for item in read_data1(get_abs_path('data')):
        rst_items += proc_confusion_item(item)
    print(len(rst_items))
    for item in read_confusion_data(get_abs_path('data')):
        rst_items += proc_confusion_item(item)
    print(len(rst_items))
    #
    # # 拆分训练与测试
    dev_set_len = len(rst_items) // 10
    # random.seed(42)
    # random.shuffle(rst_items)
    dump_json(rst_items[:dev_set_len], get_abs_path('data', 'dev.json'))
    dump_json(rst_items, get_abs_path('data', 'train.json'))
    dump_json(rst_mlt, get_abs_path('data', 'test.json'))
    gc.collect()
