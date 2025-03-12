""" Evaluation script for RAG models."""
import os
import sys

import evaluate
from typing import List, Dict, Any
import numpy as np
import re
import string
from collections import Counter

sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip


def show_all_metrics():
    metrics_list = evaluate.list_evaluation_modules()
    for metric in metrics_list:
        print(metric)


def show_detail_metric(metric_name):
    metric = evaluate.load(metric_name)
    print(metric)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def word_level_f1_score(references,predictions):
    '''
    Word Level F1 Score
    '''
    num_same_words = 0
    num_pred_words = 0
    num_ref_words = 0
    for reference, prediction in zip(references, predictions):
        prediction_words = prediction.split()
        reference_words = reference.split()
        common = Counter(prediction_words) & Counter(reference_words)
        num_same = sum(common.values())
        num_pred_words += len(prediction_words)
        num_ref_words += len(reference_words)
        num_same_words += num_same
    if num_same_words==0:
        return 0.0
    else:
        precision = 1.0 * num_same_words / num_pred_words
        recall = 1.0 * num_same_words / num_ref_words
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


class QAMetric:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.exact_match = evaluate.load('exact_match')
        self.f1 = evaluate.load('f1')
        self.sacrebleu = evaluate.load('sacrebleu')

    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Ensure predictions and references are lists of strings
        predictions = [str(p) if p is not None else "" for p in predictions]
        references = [str(r) if r is not None else "" for r in references]

        # Calculate ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True
        )
        
        # Calculate Exact Match score
        em_score = self.exact_match.compute(
            predictions=predictions,
            references=references
        )

        # Calculate F1 score
        f1_score = self.f1.compute(
            predictions=predictions,
            references=references
        )

        # Calculate BLEU score
        bleu_score = self.sacrebleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )

        return {
            'ROUGE-L': round(rouge_scores['rougeL'] * 100, 2),
            'EM': round(em_score['exact_match'] * 100, 2),
            'F1': round(f1_score['f1'] * 100, 2),
            'SacreBLEU': round(bleu_score['score'], 2)
        }

