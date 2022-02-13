
import numpy as np
from tqdm import tqdm

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]/float(len(s1))


def match_string_lists(a,b):
    best_matches = []
    matched = set()
    for word in tqdm(a):

        best_distance = np.inf
        best_match = ''

        for other_word in b:
            if other_word not in matched:
                distance = levenshtein(word, other_word)
                if distance < best_distance:
                    best_distance = distance
                    best_match = other_word

        best_matches.append((word, best_match, best_distance))
        matched.add(best_match)

    return best_matches



#Vectorized Versions
import editdistance
from itertools import product

def diffs(l):
    return [editdistance.eval(a,b)/max(len(a), len(b)) for a,b in product(l,l)]

def max_diff(l):
    '''
    For a list of strings L, returns the maximum difference (levenstein distance)
    '''
    return max(diffs(l))

