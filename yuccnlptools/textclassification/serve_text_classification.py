"""
线上服务使用
"""

import typing
import dataclasses
import time

import tqdm
import torch
import transformers
import yucctools as yt
import yuccnlptools as ynt

# from ..data.processors.genernal import SmpRankProcessor
# from ..data.processors.genernal import genernal_convert_examples_to_features


logger = yt.logger()


@dataclasses.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = dataclasses.field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: typing.Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: typing.Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: typing.Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


class PredictDataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)


class TextClassificationModel:

    def __init__(self, model_name_or_path, args=None):
        # 使用cpu或者gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 使用的模型文件
        # model_name_or_path = '/dfsdata2/yucc1_data/output/smp-rank-online'
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        # 转入device，并设置为eval
        # tokenizer.to(device)
        model.to(device)
        model.eval()

        # 获得类别列表
        label_list = ynt.SmpRankProcessor.get_labels()

        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.label_list = label_list
        self.max_seq_length = 30
        self.output_mode = 'classification'
        self.batch_size = 8

    def batch_predict(self, texts, topic):
        assert len(texts) == self.batch_size
        examples = []
        for index, text in enumerate(texts):
            guid = f'predict-{index}'
            text_a = text
            text_b = None
            label = None
            example = transformers.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        features = ynt.genernal_convert_examples_to_features(
                        examples=examples,
                        tokenizer=self.tokenizer,
                        max_length=self.max_seq_length,
                        label_list=self.label_list,
                        output_mode=self.output_mode,
                        log=False
                    )
        predict_dataset = PredictDataset(features)
       
        sampler = torch.utils.data.sampler.SequentialSampler(predict_dataset)
        dataloader = torch.utils.data.dataloader.DataLoader(
                predict_dataset,
                batch_size=self.batch_size,
                collate_fn=transformers.default_data_collator,
                drop_last=False)
        for inputs in dataloader:
            for k, v in inputs.items():
                # print(k, v)
                if isinstance(v, torch.Tensor):
                    # print(v.shape)
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                # print(logits)
                topic_index = self.label_list.index(topic)
                topic_logits = logits[:, topic_index]
        topic_logits = topic_logits.tolist()
        results = list(zip(topic_logits, texts))
        results = sorted(results, key=lambda x:x[0], reverse=True)
        return results
        # print(results)


if __name__ == '__main__':
    tcm = TextClassificationModel()
    texts = ['哈哈哈，今天一起去羽毛球、乒乓球啊', '运动, 运动我最热爱运动', '音乐', '我爱篮球', '想去打羽毛球', '测试', '今天去运动', '周杰伦']
    topic = '体育'
    start_time = time.time()
    tcm.batch_predict(texts, topic)
    print(time.time() - start_time)



            

            


