"""
Parent contamination detection class
"""

import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from llmsanitize.configs.config import supported_methods, config
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("base")


class BaseContaminationChecker:
    """ Base class of ContaminationChecker
    """

    def __init__(self, args):
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.supported_methods = supported_methods

        # download the datasets -> using HF's datasets
        self.download_data()

        # subsample the eval dataset
        self.subsample_eval_data()

        # standardize the text field
        if self.text_keys != [] and self.text_keys != ['']:
            self.combine_text_keys()
        else:
            self.normalize_text_key()

    def download_data(self):
        if self.train_data_name:
            self.train_data = load_dataset(self.train_data_name)
            self.train_data = self.train_data['train']
        else:
            self.train_data = []

        if self.eval_data_name:
            self.eval_data = load_dataset(self.eval_data_name)
            self.eval_data = self.eval_data[self.eval_set_key]
        else:
            self.eval_data = []

        logger.info(f"There are {len(self.train_data)} train data points and {len(self.eval_data)} eval data points")

    def subsample_eval_data(self):
        if len(self.eval_data) > 0 and self.n_eval_data_points > 0:
            p = np.random.permutation(len(self.eval_data))
            p = p[:self.n_eval_data_points]
            p = list(p)
            self.eval_data = self.eval_data.select(p)
            logger.info(f"After sub-sampling, there are now {len(self.eval_data)} eval data points")

    def combine_text_keys(self):
        for key in self.text_keys:
            assert key in self.train_data.features, "Error - please provide a text key that is in this dataset"
        self.train_data = self.combine_text_keys_subset_(self.train_data)
        self.eval_data = self.combine_text_keys_subset_(self.eval_data)

    def combine_text_keys_subset_(self, subset):
        texts = []
        vals = {}
        for key in self.text_keys:
            vals[key] = subset[key]
        for i in tqdm(range(len(subset))):
            text = ""
            for j in range(len(self.text_keys)):
                key = self.text_keys[j]
                if j == 0:
                    text += str(vals[key][i])
                else:
                    text += " | " + str(vals[key][i])
            texts.append(text)
        subset = subset.add_column("text", texts)
        
        return subset

    def normalize_text_key(self):
        if self.train_data:
            self.train_data = self.normalize_text_key_(self.train_data)
        if self.eval_data:
            self.eval_data = self.normalize_text_key_(self.eval_data)

    def normalize_text_key_(self, subset):
        if self.text_key != "text":
            assert self.text_key in subset.features, "Error - please provide a text key that is in this dataset"
            subset = subset.add_column("text", subset[self.text_key])
        
        return subset

    def run_contamination(self, method):
        logger.info("run_contamination not implemented")
        pass
