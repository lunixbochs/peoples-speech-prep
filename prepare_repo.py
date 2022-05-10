from pathlib import Path
import json
import os
import sys
import tarfile

from tqdm import tqdm

def make_subset(split_dir: Path, split: str) -> None:
    json_path  = split_dir / f"{split}.json"
    json0_path = split_dir / f"{split}_000000.json"
    first_tar  = split_dir / f"{split}" / f"{split}_000000.tar"

    with tarfile.open(first_tar) as tf:
        filenames = set(os.path.normpath(name) for name in tf.getnames() if name.endswith('.flac'))
        todo = filenames.copy()

    train_keys = ['duration_ms', 'label', 'name', 'raw_label']
    total = json_path.stat().st_size
    with json_path.open() as f, json0_path.open('w') as out, tqdm(total=total, desc=f"subset {split_dir.name}") as pbar:
        for line in f:
            j = json.loads(line)
            train = j['training_data']
            valid = []
            train_keys = list(train.keys())
            for values in zip(*(train[k] for k in train_keys)):
                obj = dict(zip(train_keys, values))
                name = os.path.normpath(obj['name'])
                if name in filenames:
                    todo.remove(name)
                    valid.append(values)
            if valid:
                result = j.copy()
                result['training_data'] = dict(zip(train_keys, zip(*valid)))
                out.write(json.dumps(result) + '\n')
            pbar.update(len(line))

    if todo:
        print(f"subset {split} missing {len(todo)} / {len(filenames)}")

def build_index(repo_dir: Path) -> None:
    splits = [
        ('train-clean',    'train', 'clean'),
        ('train-clean-sa', 'train', 'clean_sa'),
        ('train-dirty',    'train', 'dirty'),
        ('train-dirty-sa', 'train', 'dirty_sa'),
        ('dev',  'dev',    'dev'),
        ('test', 'test',   'test'),
    ]
    index = {
        'splits': {},
    }
    def norm_path(path: str) -> str:
        return os.path.relpath(path, repo_dir)

    for name, split_path, json_name in tqdm(splits, desc='index'):
        split_dir  = repo_dir / split_path
        json_path  = split_dir / f"{json_name}.json"
        json0_path = split_dir / f"{json_name}_000000.json"
        make_subset(split_dir, json_name)

        tar_dir = split_dir / json_name
        if not tar_dir.exists():
            tar_dir = split_dir
        tars = [str(name) for name in tar_dir.iterdir() if str(name).endswith('.tar')]
        tars.sort()

        split_data = {
            "json_full": norm_path(str(json_path)),
            "json_0":    norm_path(str(json0_path)),
            "tars":      [norm_path(name) for name in tars],
        }
        index['splits'][name] = split_data

    with (repo_dir / 'index.json').open('w') as f:
        json.dump(index, f)

if __name__ == '__main__':
    root = Path(__file__).parent
    build_index(root)
