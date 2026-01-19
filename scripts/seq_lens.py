"""

"""

import os
import sys
import json
import argparse
__author__ = 'Rob Edwards'

def run(args):
    data = json.load(open(args.f))
    for seq in data:
        print(f"{len(seq['aa_sequence'])}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('-f', help='input file', required=True)
    parser.add_argument('-v', help='verbose output', action='store_true')
    args = parser.parse_args()

    run(args)


