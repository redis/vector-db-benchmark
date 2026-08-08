#!/usr/bin/env python3
"""
Script to add missing vector_count and description fields to datasets.json
"""

import json
import re

_MAGNITUDE = {'k': 10 ** 3, 'm': 10 ** 6, 'g': 10 ** 9, 'b': 10 ** 9}


def _size_from_corpus_path(path):
    """Corpus size advertised by the corpus filename itself, or None.

    Only the LEAF path segment is considered (so a family directory like
    `yandex-1B-200-angular/` cannot override the `..._100k` corpus inside it),
    and the LAST magnitude token in it wins.
    """
    if not isinstance(path, str):
        return None
    segments = [s for s in path.split('/') if s]
    if not segments:
        return None
    size = None
    for token in re.split(r'[_\-.]', segments[-1]):
        match = re.fullmatch(r'(\d+)([kKmMgGbB])', token)
        if match:
            size = int(match.group(1)) * _MAGNITUDE[match.group(2).lower()]
    return size


def estimate_vector_count(name, path=None):
    """Estimate vector count from dataset name patterns.

    The dataset NAME is an unreliable source: `random-100-match-kw-small-vocab-*`
    is named for its 100 dimensions of query vocabulary, not its corpus size, but
    points at `random_keywords_1m_vocab_10` — a 1,000,000-point corpus. Guessing
    100 from the name is what produced the silently-wrong recall in issue #224.
    So when the CORPUS PATH names its own magnitude (`..._1m`, `..._100k`),
    that wins over anything read out of the name.
    """
    from_path = _size_from_corpus_path(path)
    if from_path is not None:
        return from_path

    name_lower = name.lower()

    # Direct patterns
    if '1b' in name_lower or '1billion' in name_lower or '1g' in name_lower:
        return 1000000000
    elif '400m' in name_lower:
        return 400000000
    elif '200m' in name_lower:
        return 200000000
    elif '100m' in name_lower:
        return 100000000
    elif '40m' in name_lower:
        return 40000000
    elif '20m' in name_lower:
        return 20000000
    elif '10m' in name_lower:
        return 10000000
    elif '1m' in name_lower:
        return 1000000
    elif '100k' in name_lower:
        return 100000
    elif '10k' in name_lower:
        return 10000
    elif '1k' in name_lower:
        return 1000
    elif 'random-100' in name_lower:
        return 100
    
    # Special cases
    if 'glove' in name_lower:
        return 1183514  # Standard GloVe size
    elif 'deep-image' in name_lower:
        return 9990000  # Standard deep image size
    elif 'gist' in name_lower:
        return 1000000  # Standard GIST size
    elif 'yandex' in name_lower and '100k' in name_lower:
        return 100000
    elif 'dbpedia' in name_lower:
        return 1000000
    elif 'h-and-m' in name_lower:
        return 105100  # measured: vectors.npy shape (105100, 2048)
    elif 'arxiv' in name_lower:
        return 2138591  # measured: vectors.npy shape (2138591, 384)
    elif 'laion-small-clip' in name_lower:
        return 100000
    elif 'random-match' in name_lower or 'random-range' in name_lower or 'random-geo' in name_lower:
        if '2048' in name_lower:
            return 100000  # 2048D synthetic datasets
        else:
            return 1000000  # 100D synthetic datasets
    elif 'random-100-match' in name_lower:
        return 100  # Small vocab datasets

    # Default for unknown patterns
    return None

def generate_description(name):
    """Generate description from dataset name patterns"""
    name_lower = name.lower()
    
    if 'laion' in name_lower:
        return 'Image embeddings'
    elif 'glove' in name_lower:
        return 'Word vectors'
    elif 'deep-image' in name_lower:
        return 'CNN image features'
    elif 'gist' in name_lower:
        return 'Image descriptors'
    elif 'dbpedia' in name_lower:
        return 'Knowledge embeddings'
    elif 'yandex' in name_lower:
        return 'Text-to-image embeddings'
    elif 'arxiv' in name_lower:
        return 'Academic paper embeddings'
    elif 'h-and-m' in name_lower:
        return 'Fashion product embeddings'
    elif 'random' in name_lower:
        if 'match' in name_lower and 'keyword' in name_lower:
            return 'Synthetic keyword matching'
        elif 'match' in name_lower and 'int' in name_lower:
            return 'Synthetic integer matching'
        elif 'range' in name_lower:
            return 'Synthetic range queries'
        elif 'geo' in name_lower:
            return 'Synthetic geo queries'
        else:
            return 'Synthetic data'
    else:
        return None

def main():
    # Read the datasets.json file
    with open('datasets/datasets.json', 'r') as f:
        datasets = json.load(f)
    
    updated_count = 0
    
    for dataset in datasets:
        updated = False
        
        # Add vector_count if missing
        if 'vector_count' not in dataset:
            vector_count = estimate_vector_count(dataset['name'], dataset.get('path'))
            dataset['vector_count'] = vector_count
            updated = True
            print(f"Added vector_count {vector_count} to {dataset['name']}")
        
        # Add description if missing
        if 'description' not in dataset:
            description = generate_description(dataset['name'])
            dataset['description'] = description
            updated = True
            print(f"Added description '{description}' to {dataset['name']}")
        
        if updated:
            updated_count += 1
    
    # Write back the updated datasets.json
    with open('datasets/datasets.json', 'w') as f:
        json.dump(datasets, f, indent=2)
    
    print(f"\nUpdated {updated_count} datasets")
    print("datasets.json has been updated with missing vector_count and description fields")

if __name__ == "__main__":
    main()
