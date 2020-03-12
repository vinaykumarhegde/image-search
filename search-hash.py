#! /usr/bin/env python

import cv2
import pickle
import argparse
from pathlib import Path
from utils import phash, compare_phash

def search_in_haystack(directory, treefile, hashfile):
    print("Reading VPTree file")
    with open(treefile, 'rb') as tf:
        tree = pickle.loads(tf.read())
    print("Reading hash file")
    with open(hashfile, 'rb') as hf:
        hashes = pickle.loads(hf.read())    
    files = []
    for ext in ["*.jpg", "*.jpeg"]:
        files.extend(Path(directory).rglob(ext))
    print("Searching {} files".format(len(files)))
    for f in files:
        h = phash.compute(cv2.imread(str(f)))
        ih = tree.get_nearest_neighbor(h)
        ihb = ih[1].tobytes()
        print("Match: {} : {}".format(str(hashes[ihb][0]), f))

parser = argparse.ArgumentParser(description="Search for image in hash/tree")
parser.add_argument('tree', help="VPTree file path")
parser.add_argument('hash', help="Hash path")
parser.add_argument('needle', help="dir with files to search for")
args = parser.parse_args()
print(args)
search_in_haystack(args.needle, args.tree, args.hash)
