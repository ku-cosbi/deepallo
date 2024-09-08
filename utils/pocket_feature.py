#!/usr/bin/env python
# -*- coding: utf-8 -*-

def pocket_feature(file_dir):
    pocket = open(file_dir + "", "r").readlines()
    pocket_num = len(pocket) // 21

    # collection of features of all pockets
    features = []

    for index in range(pocket_num):
        # feature for current pocket
        cur_feature = []
        cur = pocket[index * 21: (index + 1) * 21]
        for line in cur[1:-1]:
            cur_feature.append(float(line.split("\t")[2][:-1]))

        assert len(cur_feature) == 19
        features.append(cur_feature)

    return features
