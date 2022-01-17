""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):

    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_lens

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if isinstance(example.text_a,list):
            example.text_a = " ".join(example.text_a)
        tokens = tokenizer.tokenize(example.text_a)

        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length


        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        return ['I-inform-poi名称', 'B-inform-poi名称', 'I-deny-poi名称', 'B-deny-poi名称', 'I-inform-poi修饰',
                'B-inform-poi修饰', 'I-deny-poi修饰', 'B-deny-poi修饰', 'I-inform-poi目标', 'B-inform-poi目标',
                'I-deny-poi目标', 'B-deny-poi目标', 'I-inform-起点名称', 'B-inform-起点名称', 'I-deny-起点名称',
                'B-deny-起点名称', 'I-inform-起点修饰', 'B-inform-起点修饰', 'I-deny-起点修饰', 'B-deny-起点修饰',
                'I-inform-起点目标', 'B-inform-起点目标', 'I-deny-起点目标', 'B-deny-起点目标', 'I-inform-终点名称',
                'B-inform-终点名称', 'I-deny-终点名称', 'B-deny-终点名称', 'I-inform-终点修饰', 'B-inform-终点修饰',
                'I-deny-终点修饰', 'B-deny-终点修饰', 'I-inform-终点目标', 'B-inform-终点目标', 'I-deny-终点目标',
                'B-deny-终点目标', 'I-inform-途经点名称', 'B-inform-途经点名称', 'I-deny-途经点名称', 'B-deny-途经点名称',
                'I-inform-请求类型', 'B-inform-请求类型', 'I-deny-请求类型', 'B-deny-请求类型', 'I-inform-出行方式',
                'B-inform-出行方式', 'I-deny-出行方式', 'B-deny-出行方式', 'I-inform-路线偏好', 'B-inform-路线偏好',
                'I-deny-路线偏好', 'B-deny-路线偏好', 'I-inform-对象', 'B-inform-对象', 'I-deny-对象', 'B-deny-对象',
                'I-inform-操作',
                'B-inform-操作', 'I-deny-操作', 'B-deny-操作', 'I-inform-序列号', 'B-inform-序列号', 'I-deny-序列号',
                'B-deny-序列号', 'I-inform-页码', 'B-inform-页码', 'I-deny-页码', 'B-deny-页码',
                'I-inform-value', 'B-inform-value', 'I-deny-value', 'B-deny-value',
                'O', "[START]", "[END]"]




ner_processors = {
    "cner": CnerProcessor
}
