import os
import json
import pandas as pd


def load_metadata(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Standardize column names
    rename_map = {
        'subj': 'subject_num',
        'group': 'group_num',
        'order (IE/EI)': 'order',
        'sex': 'sex',
        'age': 'age',
    }
    df = df.rename(columns=rename_map)
    # Keep only relevant columns
    keep_cols = [c for c in ['subject_num', 'group_num', 'order', 'sex', 'age'] if c in df.columns]
    return df[keep_cols].copy()


def _map_group(group_num: int | float) -> str:
    try:
        group_int = int(group_num)
    except Exception:
        return 'n/a'
    return 'control' if group_int == 1 else 'risk'


def _pad_subject_num(n: int | float | str) -> str:
    try:
        ni = int(float(n))
    except Exception:
        return str(n)
    return f"{ni:02d}"


def _map_incl_excl(order: str) -> tuple[str, str]:
    """Return numeric task identifiers based on order.

    IE: inclusion=2, exclusion=4
    EI: inclusion=4, exclusion=2
    """
    if isinstance(order, str) and order.upper() == 'IE':
        return '2', '4'
    if isinstance(order, str) and order.upper() == 'EI':
        return '4', '2'
    return 'n/a', 'n/a'


def write_participants(raw_root: str, subjects_list: list[str], tasks: list[str], metadata_df: pd.DataFrame) -> None:
    os.makedirs(raw_root, exist_ok=True)
    participants_tsv = os.path.join(raw_root, 'participants.tsv')
    participants_json = os.path.join(raw_root, 'participants.json')

    subjects_set = {str(s) for s in subjects_list}

    rows = []
    for _, row in metadata_df.iterrows():
        subj_padded = _pad_subject_num(row.get('subject_num'))
        if subj_padded not in subjects_set:
            continue
        group_text = _map_group(row.get('group_num'))
        order = row.get('order') if isinstance(row.get('order'), str) else 'n/a'
        inclusion_task, exclusion_task = _map_incl_excl(order)
        rows.append({
            'participant_id': f"sub-{subj_padded}",
            'age': row.get('age'),
            'sex': row.get('sex'),
            'group': group_text,
            'order': order,
            'inclusion_task': inclusion_task,
            'exclusion_task': exclusion_task,
            'hand': 'n/a',
            'weight': 'n/a',
            'height': 'n/a',
        })

    import pandas as pd  # local import for safety
    out_df = pd.DataFrame(
        rows,
        columns=[
            'participant_id', 'age', 'sex', 'group', 'order',
            'inclusion_task', 'exclusion_task', 'hand', 'weight', 'height'
        ]
    )
    out_df.to_csv(participants_tsv, sep='\t', index=False)

    sidecar = {
        'participant_id': {'Description': 'Unique participant identifier'},
        'age': {'Description': 'Age in years', 'Units': 'years'},
        'sex': {'Description': 'Biological sex', 'Levels': {'M': 'male', 'F': 'female'}},
        'group': {'Description': 'Study group', 'Levels': {'control': 'healthy controls', 'risk': 'risk of depression'}},
        'order': {'Description': 'Task order indication (IE: inclusion before exclusion; EI: exclusion before inclusion)'},
        'inclusion_task': {'Description': 'Numeric task identifier used for inclusion (2=Sart2, 4=Sart4)'},
        'exclusion_task': {'Description': 'Numeric task identifier used for exclusion (2=Sart2, 4=Sart4)'},
        'hand': {'Description': 'Handedness', 'Levels': {'R': 'right', 'L': 'left', 'A': 'ambidextrous', 'n/a': 'not available'}},
        'weight': {'Description': 'Self-reported weight', 'Units': 'kg'},
        'height': {'Description': 'Self-reported height', 'Units': 'cm'},
    }
    with open(participants_json, 'w') as f:
        json.dump(sidecar, f, indent=2)


