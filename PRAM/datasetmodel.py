import enum
import json

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random
from typing import Any, Optional, cast
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ArrayDict = dict[str, np.ndarray]
Info = dict[str, Any]
DATA_DIR = Path.home() / 'repositories' / 'a' / 'data'
SEED = 0
CAT_MISSING_VALUE = '__nan__'

class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'

def _set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)

def _make_split(size: int, stratify: Optional[np.ndarray], n_parts: int) -> ArrayDict:
    # n_parts == 3:  train & val & test
    # n_parts == 2:  train & val
    assert n_parts in (2, 3)
    all_idx = np.arange(size, dtype=np.int64)
    a_idx, b_idx = train_test_split(
        all_idx,
        test_size=0.1,
        stratify=stratify,
        random_state=SEED + (1 if n_parts == 2 else 0),
    )
    if n_parts == 2:
        return cast(ArrayDict, {'train': a_idx, 'val': b_idx})
    a_stratify = None if stratify is None else stratify[a_idx]
    a1_idx, a2_idx = train_test_split(
        a_idx, test_size=0.1, stratify=a_stratify, random_state=SEED + 1
    )
    return cast(ArrayDict, {'train': a1_idx, 'val': a2_idx, 'test': b_idx})

def _apply_split(data: ArrayDict, split: ArrayDict) -> dict[str, ArrayDict]:
    return {k: {part: v[idx] for part, idx in split.items()} for k, v in data.items()}

def _encode_classification_target(y: np.ndarray) -> np.ndarray:
    assert not str(y.dtype).startswith('float')
    if str(y.dtype) not in ['int32', 'int64', 'uint32', 'uint64']:
        y = LabelEncoder().fit_transform(y)
    else:
        labels = set(map(int, y))
        if sorted(labels) != list(range(len(labels))):
            y = LabelEncoder().fit_transform(y)
    return y.astype(np.int64)

def _save(
    dataset_dir: Path,
    name: str,
    task_type: TaskType,
    *,
    X_num: Optional[ArrayDict],
    X_cat: Optional[ArrayDict],
    y: ArrayDict,
    idx: Optional[ArrayDict],
    id_: Optional[str] = None,
    id_suffix: str = '--default',
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
        X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}
    if idx is not None:
        idx = {k: v.astype(np.int64) for k, v in idx.items()}
    y = {
        k: v.astype(np.float32 if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
        'name': name,
        'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
        'task_type': task_type.value,
        'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
        'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
    } | {f'{k}_size': len(v) for k, v in y.items()}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y', 'idx']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)
    (dataset_dir / 'READY').touch()
    print('Done\n')


def dropout_modelling():

    dataset_dir = Path.cwd() / 'data' / 'dropout' / 'pramnpy'
    df = pd.read_csv('data/dropout/pramdropout_e2.csv')

    y_all = _encode_classification_target(df.pop('Target').values)
    num_columns = df.columns.tolist()

    assert set(num_columns)  == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float32).values

    idx = _make_split(len(df), y_all, 3)

    _save(
        dataset_dir,
        'Dropout Modelling',
        TaskType.MULTICLASS,
        **_apply_split(
            {'X_num': X_num_all, 'y': y_all},
            idx,
        ),
        X_cat=None,
        idx=idx,
    )

def ukm_modelling():

    dataset_dir = Path.cwd() / 'data' / 'ukm' / 'pramnpy'
    df = pd.read_csv('data/ukm/pramukm.csv')

    y_all = df.pop('UNS').values

    num_columns = df.columns.tolist()

    X_num_all = df[num_columns].astype(np.float32).values

    idx = _make_split(len(df), None, 3)

    _save(
        dataset_dir,
        'UKM Modelling',
        TaskType.REGRESSION,
        **_apply_split(
            {'X_num': X_num_all, 'y': y_all},
            idx,
        ),
        X_cat=None,
        idx=idx,
    )


ukm_modelling()