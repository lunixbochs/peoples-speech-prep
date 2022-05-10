from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Tuple
import argparse
import io
import itertools
import os
import sys
import tarfile

from tqdm import tqdm

def tqdm_bytes(*args, **kwargs):
    return tqdm(*args, **kwargs, unit='B', unit_scale=True, unit_divisor=1024)

def tqdm_file(path: str) -> tqdm:
    name    = os.path.basename(path)
    tarsize = os.stat(path).st_size
    return tqdm_bytes(desc=name, total=tarsize)

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
def resplit_tars(name: str, src: Path, dst: Path, split_max: int) -> Iterator[str]:
    dst.mkdir(exist_ok=True, parents=True)
    if os.path.isfile(src):
        src_tars = [src]
    else:
        src_tars = sorted([ent.path for ent in os.scandir(src) if ent.name.endswith('.tar')])
    target_size = split_max - tarfile.BLOCKSIZE

    info = fileobj = None
    src_iter = iter_tars(name, src_tars)
    for n in itertools.count():
        out_path = str(dst / f"{name}_{n:06d}.tar")
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

def parse_size(size: str) -> int:
    suffixes = {c: 1024 ** i for i, c in enumerate('BKMGT')}
    coeff = 1
    for c, cmul in suffixes.items():
        if size.endswith(c):
            size = size[:-1]
            coeff = cmul
            break
    return int(size) * coeff

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', help='source file or directory for original tars')
    parser.add_argument('dst', help='destination directory for split tars')
    parser.add_argument('--prefix', help='naming prefix for output files', default=None)
    parser.add_argument('--maxsize', help='maximum size per tar file (supports suffixes B, K, M, G, T)', default='100M')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    maxsize = parse_size(args.maxsize)
    prefix = args.prefix or src.name.split('.', 1)[0]
    for tar in resplit_tars(prefix, src, dst, split_max=maxsize):
        pass
