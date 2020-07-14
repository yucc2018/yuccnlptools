"""
通用的数据processor
"""
import os
import json
import typing

import tqdm
import sklearn
import sklearn.model_selection
import transformers
import yucctools as yt


logger = yt.logger()


def genernal_convert_examples_to_features(
        examples,
        tokenizer,
        max_length = None,
        label_list = None,
        output_mode = None,
        log = True,
    ):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: transformers.InputExample) -> typing.Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        # print(output_mode)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = transformers.InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    if log:
        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features[i])

    return features


def data_process(data):
    blocklist = []
    with open('blocklist', 'r') as f:
        for line in f:
            text = line.strip().split("'")[1]
            blocklist.append(text)

    examples = []
    for topic, session_list in data.items():
        for session in session_list:
            # 楼主 姓名和内容
            user = session.get('topic').get('name')
            content = session.get('title')
            # 加入主题帖，使用标题，而不是内容
            post = {'user': session.get('topic').get('name'),
                   'date': session.get('topic').get('date'),
                    'content': session.get('title')}
            # copy 深度拷贝
            replys = session.get('replys').copy()
            replys.insert(0, post)

            # 2句对话，3句对话，4句对话，5句对话的内容
            for context_len in [2, 3, 4, 5]:
                replys_len = len(replys)
                if context_len < replys_len:
                    for start in range(0, replys_len-context_len):
                        # 获得内容；去除换行符；加入主题内容
                        cur_pairs = [r.get('content') for r in replys[start: start+context_len]]
                        cur_pairs = [p.replace('\n', ' ').strip() for p in cur_pairs]
                        cur_pairs = [topic] + cur_pairs

                        cur_pairs = [p.strip() for p in cur_pairs]
                        response = cur_pairs[-1]
                        if len(response) <= 2 or len(response) >= 30 or response in blocklist:
                            continue
                        else:
                            examples.append(cur_pairs)
    return examples


class SmpLMProcessor:

    def __init__(self, args):
        train_data = {}
        test_data = {}
        # print(args.data_dir)
        for topic in os.listdir(args.data_dir):
            # 加载数据
            topic_path = os.path.join(args.data_dir, topic)
            if not os.path.isdir(topic_path) or topic == '.ipynb_checkpoints':
                continue
            logger.info(f'topic: {topic}')
            sess_list = []
            for file in tqdm.tqdm(os.listdir(topic_path)):
                with open(os.path.join(topic_path, file), 'r', encoding='utf-8') as f:
                    sess = json.load(f)
                    sess['pid'] = file.split('.')[0]
                    sess_list.append(sess)
            # 从源头切割数据
            train_sess_list, test_sess_list = sklearn.model_selection.train_test_split(sess_list,
                test_size=0.05, random_state=1)
            train_data[topic] = train_sess_list
            test_data[topic] = test_sess_list
        # 转换成examples
        train_examples = data_process(train_data)
        test_examples = data_process(test_data)
        all_examples = train_examples + test_examples
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.all_examples = all_examples


def _smp_rank_data_process(data, args, set_type):
    logger.info(f'block dir: {args.block_dir}')
    blocklist = []
    if args.block_dir:
        with open(args.block_dir, 'r') as f:
            for line in f:
                text = line.strip().split("'")[1]
                blocklist.append(text)

    final_data = []
    for topic, session_list in data.items():
        for session in session_list:
            # 楼主 姓名和内容
            user = session.get('topic').get('name')
            content = session.get('title')
            # 加入主题帖，使用标题，而不是内容
            post = {'user': session.get('topic').get('name'),
                   'date': session.get('topic').get('date'),
                    'content': session.get('title')}
            # copy 深度拷贝
            replys = session.get('replys').copy()
            replys.insert(0, post)

            for index, reply in enumerate(replys):
                text_a = reply.get('content')
                text_a = text_a.replace('\n', '').strip()
                if len(text_a) <= 2 or len(text_a) >= 30 or text_a in blocklist:
                    continue
                else:
                    guid = f'{set_type}-{index}' 
                    example = transformers.InputExample(guid=guid, text_a=text_a, text_b=None, label=topic)
                    final_data.append(example)
    return final_data


class SmpRankProcessor(transformers.DataProcessor):
    def __init__(self, args):
        train_data = {}
        test_data = {}
        # print(args.data_dir)
        for topic in os.listdir(args.data_dir):
            # 加载数据
            topic_path = os.path.join(args.data_dir, topic)
            if not os.path.isdir(topic_path) or topic == '.ipynb_checkpoints':
                continue
            logger.info(f'topic: {topic}')
            sess_list = []
            for file in tqdm.tqdm(os.listdir(topic_path)):
                with open(os.path.join(topic_path, file), 'r', encoding='utf-8') as f:
                    sess = json.load(f)
                    sess['pid'] = file.split('.')[0]
                    sess_list.append(sess)
            # 从源头切割数据
            train_sess_list, test_sess_list = sklearn.model_selection.train_test_split(sess_list,
                test_size=0.05, random_state=1)
            train_data[topic] = train_sess_list
            test_data[topic] = test_sess_list
        # 转换成examples
        train_examples = _smp_rank_data_process(train_data, args, set_type='train')
        test_examples = _smp_rank_data_process(test_data, args, set_type='test')
        all_examples = train_examples + test_examples
        if args.online:
            self.train_examples = all_examples
        else:
            self.train_examples = train_examples
        self.test_examples = test_examples
        self.all_examples = all_examples

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.test_examples

    def get_test_examples(self):
        return self.test_examples

    @classmethod
    def get_labels(cls):
        return ['体育', '数码产品', '电影', '美食', '音乐']

genernal_tasks_num_labels = {
    'smp-rank': 5,
}

genernal_processors = {
    "smp-rank": SmpRankProcessor,
}

genernal_output_modes = {
    "smp-rank": "classification",
}
