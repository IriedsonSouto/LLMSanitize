"""
This file implements the string-matching done for data contamination as in GPT-4's paper.
"""

from llmsanitize.utils.string_utils import *
from llmsanitize.utils.logger import get_child_logger

logger = get_child_logger("gpt4")


def clean_text_gpt4(text):
    text = ''.join(i if i.isalpha() else '' for i in text)  # keep alphanumeric characters

    return text

# Following the logic in GPT-4's report: https://arxiv.org/pdf/2303.08774.pdf appendix C
def main_gpt4(
    train_data,
    eval_data,
    train_data_name,
    eval_data_name,
    eval_set_key
):
    ## only keep the content per data example, discard labels
    train_data = train_data["text"]
    eval_data = eval_data["text"]

    string_size = 50
    train_strings = build_strings(train_data, string_size, clean_text_gpt4)
    logger.info(f"There are {len(train_strings.keys())} {string_size}-chars strings in the training set")

    n_samples = 3
    contaminated = overlap_strings_sample(eval_data, train_strings, string_size, n_samples, clean_text_gpt4)
    frac = 100 * np.mean(contaminated)
    n_contaminated = np.sum(contaminated)
    logger.info(f"Data contamination: checking {eval_data_name}/{eval_set_key} against {train_data_name} (train)")
    logger.info(f"Method: sampling {n_samples} {string_size}-chars substring (GPT-4 style data contamination)")
    logger.info(f"# Contaminated points: {n_contaminated}/{len(contaminated)} or {frac:.4f}%")
