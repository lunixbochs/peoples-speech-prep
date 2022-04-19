from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Tuple
import hashlib
import io
import itertools
import math
import os
import re
import sys
import tarfile

import json

from tqdm import tqdm

def rename_split(name: str) -> str:
    by_sa = 'by_sa' in name
    name = name.rsplit('_', 1)[-1]
    return f"{name}_sa" if by_sa else name

SPLIT_MAX = 100 * 1024 * 1024 # 100MB
SPLITS  = ['cc_by_clean', 'cc_by_dirty', 'cc_by_sa_clean', 'cc_by_sa_dirty']
RENAMES = {name: rename_split(name) for name in SPLITS}

def tqdm_bytes(*args, **kwargs):
    return tqdm(*args, **kwargs, unit='B', unit_scale=True, unit_divisor=1024)

def tqdm_file(path: str) -> tqdm:
    name    = os.path.basename(path)
    tarsize = os.stat(path).st_size
    return tqdm_bytes(desc=name, total=tarsize)

def hash_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(0x10000)
            h.update(chunk)
            if not chunk:
                break
    return h.hexdigest()

# iterate over the members of a tar file for reading, with a progress bar
def tqdm_tar_iter(path: str) -> Iterator[Tuple[tarfile.TarInfo, BinaryIO]]:
    with open(path, 'rb+') as f:
        with tarfile.open(fileobj=f) as tf, tqdm_file(path) as pbar:
            while True:
                info = tf.next()
                if not info: break
                yield info, tf.extractfile(info)
                pbar.update(512 + info.size)

# iterate over a list of tar files, with data and file level progress
def iter_tars(name: str, paths: List[str]) -> Iterator[Tuple[tarfile.TarInfo, BinaryIO]]:
    total_tar_size = sum([os.stat(path).st_size - 1024 for path in paths])
    with tqdm_bytes(total=total_tar_size) as pbar:
        for i, path in enumerate(paths, start=1):
            pbar.set_description(f"split {name} ({i}/{len(paths)})")
            for info, fileobj in tqdm_tar_iter(path):
                yield info, fileobj
                pbar.update(info.size + 512)

# read every tar in a directory, combine them, and split the result into many new tars, each smaller than split_max
def resplit_tars(split: str, src: Path, dst: Path, file_count: int, split_max: int) -> Iterator[str]:
    dst.mkdir(exist_ok=True, parents=True)
    src_tars = sorted([ent.path for ent in os.scandir(src) if ent.name.endswith('.tar')])
    target_size = split_max - tarfile.BLOCKSIZE

    info = fileobj = None
    src_iter = iter_tars(split, src_tars)
    for n in itertools.count():
        out_path = str(dst / f"{split}_{n:06d}.tar")
        with tarfile.open(out_path, 'w') as tf:
            if info is not None and fileobj is not None:
                tf.addfile(info, fileobj)

            for info, fileobj in src_iter:
                size = 512 + info.size
                if tf.offset + size > target_size:
                    break
                tf.addfile(info, fileobj)
            else:
                break
            yield out_path
    yield out_path

# count the number of clips in a peoples_speech json file
def json_count_clips(path: str) -> int:
    total = 0
    with open(path) as f, tqdm_file(path) as pbar:
        for line in f:
            j = json.loads(line)
            total += len(j['training_data']['name'])
            pbar.update(len(line))
    return total

def build_subsets(data_dir: Path) -> None:
    for split in tqdm(RENAMES.values(), desc='split subsets'):
        json_path  = data_dir / f"{split}.json"
        json0_path = data_dir / f"{split}_000000.json"
        first_tar  = data_dir / f"{split}" / f"{split}_000000.tar"

        with tarfile.open(first_tar) as tf:
            filenames = set(tf.getnames())
            todo = filenames.copy()

        train_keys = ['duration_ms', 'label', 'name', 'raw_label']
        with json_path.open() as f, json0_path.open('w') as out:
            for i, line in enumerate(f):
                j = json.loads(line)
                train = j['training_data']
                valid = []
                for duration, label, name, raw_label in zip(*(train[k] for k in train_keys)):
                    if name in filenames:
                        todo.remove(name)
                        valid.append((duration, label, name, raw_label))
                if valid:
                    result = j.copy()
                    result['training_data'] = dict(zip(train_keys, zip(*valid)))
                    out.write(json.dumps(result) + '\n')

        if todo:
            print(f"[!] subset {split} missing {len(todo)}", file=sys.stderr)

def build_manifest(repo_dir: Path, data_dir: Path) -> None:
    manifest = {
        "splits": {},
    }
    def norm_path(path: str) -> str:
        return os.path.relpath(path, repo_dir)

    for split in tqdm(RENAMES.values(), desc='split manifests'):
        json_path  = data_dir / f"{split}.json"
        json0_path = data_dir / f"{split}_000000.json"
        first_tar  = data_dir / f"{split}" / f"{split}_000000.tar"
        split_path = data_dir / split
        names = [str(name) for name in split_path.iterdir()]
        hashes = {}
        for name in tqdm(names + [str(json_path), str(json0_path)], desc='hashing'):
            hashes[norm_path(name)] = hash_path(name)
        manifest["splits"][split] = {
            "json_full":  norm_path(str(json_path)),
            "json_first": norm_path(str(json0_path)),
            "tars": [norm_path(name) for name in names],
            "hashes": hashes,
        }
    with (repo_dir / 'manifest.json').open('w') as f:
        json.dump(manifest, f)

def do_tar_splits(root: Path, data_dir: Path) -> None:
    for split in SPLITS:
        split_json = root / f"{split}.json"
        tar_dir    = root / split
        if not os.path.isfile(split_json):
            raise FileNotFoundError(split_json)
        if not os.path.isdir(tar_dir):
            raise FileNotFoundError(tar_dir)

    for split in SPLITS:
        name = RENAMES[split]
        split_json = root / f"{split}.json"
        src_dir    = root / split
        dst_dir    = data_dir / name
        # count = json_count_clips(str(split_json))
        count = 0

        for tar_path in resplit_tars(split=name, src=src_dir, dst=dst_dir, file_count=count, split_max=SPLIT_MAX):
            pass

        dst_json = data_dir / f"{name}.json"
        with split_json.open('rb') as f, tqdm_file(str(split_json)) as pbar, dst_json.open('wb') as o:
            while True:
                chunk = f.read(0x10000)
                if not chunk:
                    break
                o.write(chunk)
                pbar.update(len(chunk))

if __name__ == '__main__':
    root = Path('.')
    repo_dir = root / "repo_out"
    data_dir = repo_dir / "data"

    do_tar_splits(root, data_dir)
    build_subsets(data_dir)
    build_manifest(repo_dir, data_dir)
