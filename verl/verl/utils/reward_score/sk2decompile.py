import torch
import tempfile
import multiprocessing as mp
import subprocess
import os
import random


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    solution_str: The solution string to be evaluated.
    ground_truth: The ground truth answer for comparison.
    """
    type_score_value,type_num = type_score(solution_str, ground_truth, extra_info)
    compileable_score_value = compileable_score(solution_str, ground_truth, extra_info)

    if compileable_score_value == 0.0:
        return 0.0

    return type_score_value + compileable_score_value

    return 0

import re

def type_score(solution_str, ground_truth, extra_info=None):
    """
    Compute the type score based on the solution string and ground truth.
    
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
    
    Returns:
        float: Type score between 0 and 1.
    """
    # Define regex patterns to match func*, type*, var*, and field*
    patterns = [r'\bfunc\w*\b', r'\btype\w*\b', r'\bvar\w*\b', r'\bfield\w*\b']

    # Function to extract matches from a string based on the patterns
    def extract_terms(text):
        terms = set()
        for pattern in patterns:
            terms.update(re.findall(pattern, text))
        return terms

    # Extract terms from solution_str and ground_truth
    solution_terms = extract_terms(solution_str)
    ground_truth_terms = extract_terms(ground_truth)

    # Calculate the intersection (common terms) and the union (all terms)
    intersection = solution_terms.intersection(ground_truth_terms)
    union = solution_terms.union(ground_truth_terms)

    # Compute Jaccard similarity (intersection / union)
    jaccard_similarity = len(intersection) / len(union) if union else 0.0

    return jaccard_similarity, len(solution_terms) + len(ground_truth_terms)


def compileable_score(solution_str, ground_truth, extra_info=None):
    """
    Compute whether the solution string is compileable.
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
    Returns:
        float: 1.0 if the solution is compileable, 0.0 otherwise.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            source_file = os.path.join(tmpdir, "temp.c")    
            object_file = os.path.join(tmpdir, "temp.o")
            header = extra_info.get('header', '') if extra_info else ''

            # Write the solution string to a temporary C source file
            with open(source_file, 'w') as f:
                f.write(f'{header}\n\n{solution_str}')
            
            proc = subprocess.run(
                ['gcc', '-c', source_file, '-o', object_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,  # Set a timeout to avoid hanging
                check=True
            )
            if proc.returncode == 0:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
