"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from transformers import AutoTokenizer

from model import _DEFAULT_MODEL, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)

def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The implemented strategy performs a stratified 10-fold shuffle split
    accounting for the presence of particular special tokens in the truncated sequence.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    stratification_data = []
    for idx, row in df.iterrows():
        text = row['prompt'] + row['response']
        s = (
            row['label'],
            'assistant' in tokenizer.tokenize(text)[:MAX_LENGTH],
            '<|endoftext|>' in tokenizer.tokenize(text)[:MAX_LENGTH]
        )
        stratification_data.append(s)
    
    sd_to_sc = {}
    sc_to_label = {}
    for sc, xxx in enumerate(list(dict.fromkeys(stratification_data))):
        sd_to_sc[xxx] = sc
        sc_to_label[sc] = int(xxx[0])
    
    groups = np.array([sd_to_sc[sd] for sd in stratification_data])

    idx = np.arange(len(y))

    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size+val_size, random_state=random_state)

    splits = []

    for train_idx, valtest_idx in sss.split(idx, groups):
        val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, stratify=groups[valtest_idx], random_state=random_state)
        splits.append([train_idx, val_idx, test_idx])
        splits.append([train_idx, test_idx.copy(), val_idx.copy()])
    
    return splits