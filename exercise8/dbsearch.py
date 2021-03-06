#!/bin/python3
# file containing the code submitted last week, updated to correctly calculate masses
from scripts import *
import pandas as pd

def match_spectrum(wtlist, peaks, ppm=20):
    # matches a weight list as generated by fragmass with a df of peaks identified by call_peaks

    # precalculate tolerances
    low_thresh = 1 - ppm/1000000
    up_thresh = 1 + ppm/1000000
    
    matching = 0
    # init two sorted iterators across the two lists
    # uses a sorted-join-algorithm
    iterwts = iter(wtlist)
    curwt = next(iterwts)
    wtlist = sorted(wtlist)
    try:
        for _, row in peaks.sort_values(by='m/z').iterrows():
            #print(row['m/z'], curwt)
            if curwt*low_thresh < row['m/z'] < curwt*up_thresh:
                matching += 1
                curwt = next(iterwts)
            elif curwt*up_thresh < row['m/z']:
                curwt = next(iterwts)
    except StopIteration:
        pass

    return matching


def get_tryptic_peptide(aaseq: str, peaks: pd.DataFrame):
    # returns the tryptic peptide of the protein specified by aaseq best fitting the spectrum, supplied as a pandas DF
    tryptics = trypticdigest(aaseq, minwt=0)

    maxp = 0
    maxtr = None
    for tr in tryptics:
        p = match_spectrum(fragmass(tr), peaks)
        if p > maxp:
            maxp = p
            maxtr = tr

    return maxp, maxtr

if __name__ == '__main__':
    import sys
    df = pd.read_csv(sys.argv[2])
    seq = 'LHVPLEAGVVLLFK'
    print("matching", seq, "to", sys.argv[2])
    print(match_spectrum(fragmass(seq), df, ppm=20))

    print("identifying best-matching peptide from", sys.argv[1], "i", sys.argv[2])
    with open(sys.argv[1], 'r') as f:
        f.readline() # skip fasta header
        seq = ''.join([l.strip() for l in f])
    print(get_tryptic_peptide(seq, df))


