"""
线上服务使用
"""

import dataclasses

import torch
import transformers
import yucctools as yt
import yuccnlptools as ynt


logger = yt.logger()


@dataclasses.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = dataclasses.field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


class TextClassificationModel:

    def __init__(self, args=None):
        # 使用cpu或者gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 使用的模型文件
        model_name_or_path = '/dfsdata2/yucc1_data/output/smp-rank-online'
        tokenizer = transformers.BertTokenizer.from_pretrained(model_name_or_path)
        model = transformers.GPT2ModelForSequenceClassification.from_pretrained(model_name_or_path)
        # 转入device，并设置为eval
        tokenizer.to(device)
        model.to(device)
        model.eval()

        # 获得类别列表
        label_list = ynt.SmpRankProcessor.get_labels()

        self.tokenizer = tokenizer
        self.model = model
        self.label_list = label_list

    def  predict(self, text):

        with torch.no_grad():


