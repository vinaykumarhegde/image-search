#! /usr/bin/env python

import os
import cv2
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path
from vptree import VPTree
from progress.bar import Bar
from utils import compare_phash, phash

def process_haystack(directory, treefile, hashfile):
    if not os.path.exists(directory):
        print("Directory {} does not exist".format(directory))
        return False
    files = []
    for ext in ["*.jpg", "*.jpeg"]:
        files.extend(Path(directory).rglob(ext))
    hashes = dict()
    print("Calculating hashes in haystack")
    bar = Bar(message='Caclulating hashes: ', max=len(files))
    for f in files:
        h = phash.compute(cv2.imread(str(f)))
        hh = h.tobytes()
        l = hashes.get(hh, [])
        l.append(f)
        hashes[hh] = l
        bar.next()
    print()
    print("Building VPTree")
    phashes = [np.frombuffer(k, dtype=np.uint8) for k in hashes.keys()]
    tree = VPTree(phashes, compare_phash)
    print("Saving VPTree")
    with open(treefile,'wb') as tf:
        tf.write(pickle.dumps(tree))
    print("Saving Hash")
    with open(hashfile, 'wb') as hf:
        hf.write(pickle.dumps(hashes))
    

parser = argparse.ArgumentParser(description="Search image in directories")
parser.add_argument("haystack", help="Parent directory")
parser.add_argument("tree", help="tree file location")
parser.add_argument("hash", help="hash file location")
args = parser.parse_args()
process_haystack(args.haystack, args.tree, args.hash)
