import logging
import os
import time
import dataclasses
import enum
import typing

import torch
from filelock import FileLock
import transformers
# import yuccnlptools as ynt

from ..processors import genernal_processors
from ..processors import genernal_output_modes
from ..processors import genernal_convert_examples_to_features


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GenernalDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = dataclasses.field(metadata={"help": "The name of the task to train on: " + ", ".join(transformers.glue_processors.keys())})
    data_dir: str = dataclasses.field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."})
    max_seq_length: int = dataclasses.field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },)
    overwrite_cache: bool = dataclasses.field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"} )
    block_dir: str = dataclasses.field(
        default='', metadata={"help": "used for smp"},)
    online: bool = dataclasses.field(
        default=False, metadata={'help': 'default false for offline train/eval/test; online will use all data'})
 
    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(enum.Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GenernalDataset(torch.utils.data.dataset.Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GenernalDataTrainingArguments
    output_mode: str
    features: typing.List[transformers.InputFeatures]

    def __init__(
        self,
        args,
        tokenizer: transformers.PreTrainedTokenizer,
        limit_length: typing.Optional[int] = None,
        mode: typing.Union[str, Split] = Split.train,
        cache_dir: typing.Optional[str] = None,
    ):
        if args.task_name in ['smp-rank']:
            self.processor = genernal_processors[args.task_name](args)
            self.output_mode = genernal_output_modes[args.task_name]
        else:
            raise Error('task name error')
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        self.label_list = self.processor.get_labels()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                
                if args.task_name in ['smp-rank']:
                    if mode == Split.dev:
                        examples = self.processor.get_dev_examples()
                    elif mode == Split.test:
                        examples = self.processor.get_test_examples()
                    elif mode == Split.train:
                        examples = self.processor.get_train_examples()
                    else:
                        raise Error('wrong mode')

                    if limit_length is not None:
                        examples = examples[:limit_length]
        
                    features = genernal_convert_examples_to_features(
                            examples=examples,
                            tokenizer=tokenizer,
                            max_length=args.max_seq_length,
                            label_list=label_list,
                            output_mode=self.output_mode,
                    )

                self.features = features
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> transformers.InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

