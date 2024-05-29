import torch, random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from itertools import chain
from typing import List


from .utils import load_json


class CorrectorDataset(Dataset):
    def __init__(self, fp):
        self.data = load_json(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']

class BlockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, sort_key, sort_bs_num=None, shuffle=True, **kwargs):
        """
        初始化函数，继承DataLoader类
        Args:
            dataset: Dataset类的实例，其中中必须包含dataset变量，并且该变量为一个list
            sort_key: 排序函数，即使用dataset元素中哪一个变量的长度进行排序
            sort_bs_num: 排序范围，即在多少个batch_size大小内进行排序，默认为None，表示对整个序列排序
            is_shuffle: 是否对分块后的内容，进行随机打乱，默认为True
            **kwargs:
        """
        assert isinstance(dataset.data, list), "dataset为Dataset类的实例，其中中必须包含dataset变量，并且该变量为一个list"
        super().__init__(dataset, **kwargs)
        self.sort_bs_num = sort_bs_num
        self.sort_key = sort_key
        self.shuffle = shuffle

    def __iter__(self):
        self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num,
                                               self.sort_key, self.shuffle)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)


    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, shuffle):
        # 将数据按照batch_size大小进行切分
        tail_data = [] if len(data) % batch_size == 0 else data[-(len(data) % batch_size):]
        data = data[:len(data) - len(tail_data)]
        assert len(data) % batch_size == 0
        # 获取真实排序范围
        sort_bs_num = len(data) // batch_size if sort_bs_num is None else sort_bs_num
        # 按照排序范围进行数据划分
        data = [data[i:i + sort_bs_num * batch_size] for i in range(0, len(data), sort_bs_num * batch_size)]
        # 在排序范围，根据排序函数进行降序排列
        data = [sorted(i, key=sort_key, reverse=True) for i in data]
        # 将数据根据batch_size获取batch_data
        data = list(chain(*data))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # 判断是否需要对batch_data序列进行打乱
        if shuffle:
            random.shuffle(data)
        # 将tail_data填补回去
        data = list(chain(*data)) + tail_data
        return data

def get_corrector_loader(fp, tokenizer, sort_key, **kwargs):
    def _collate_fn(data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [tokenizer.tokenize(t) for t in ori_texts]
        max_len = min(max([len(t) for t in encoded_texts]) + 2, 180)
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                if idx >= max_len:
                    continue
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) or encoded_text[idx + move].startswith(
                        '##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels

    dataset = CorrectorDataset(fp)
    loader = BlockShuffleDataLoader(dataset, sort_key, collate_fn=_collate_fn, **kwargs)
    return loader
