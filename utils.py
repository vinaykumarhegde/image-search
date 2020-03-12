#! /usr/bin/env python

import cv2
import numpy as np

phash = cv2.img_hash.PHash_create()

def compare_phash(p1, p2):
    return phash.compare(p1.reshape(1,8), p2.reshape(1,8))
