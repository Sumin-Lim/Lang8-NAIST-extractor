'''
Description: This file preprocesses and make Lang-8 dataset.

Usage: python preprocessing.py -d [file path] -o [file path]
-d: pickled result file path after running extract_err-cor-pair.py
-o: file path to save the result of this script
'''
import sys 
sys.path.insert(1, '../KoBERT/')
import pickle as pkl
from tqdm import tqdm 
import pandas as pd
import re
import argparse

import math
import torch
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from levenshtein import *
from hangul_util import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', nargs=1, type=str, default='../data/corrected_sentence_pair.pkl')
    parser.add_argument('-o', '--output', nargs=1, type=str, default='../data/korean_lang8.csv')
    args = parser.parse_args()
    return args

def load_data(file_path):
    original = []
    corrected = []
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
        for line in tqdm(data):
            o, c = line.split('\t')
            original.append(o)
            corrected.append(c)

    df = pd.DataFrame({'original': original, 'corrected': corrected})
    return df

def drop_other_lang(df):
    hangul = re.compile('[^ ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z@#$%&\(\){|}\[\]\/\-><=_:;.,?!~\^"（）\'‘’]+')
    df['o_hangul_len'] = df['original'].apply(lambda x: len(list(set(hangul.findall(x)))))
    df['c_hangul_len'] = df['corrected'].apply(lambda x: len(list(set(hangul.findall(x)))))
    
    df = df[(df['o_hangul_len']==0) & (df['c_hangul_len']==0)][['original', 'corrected']]
    return df

def calc_vars(df):
    df['jamo_levenshtein'] = df.apply(lambda row: jamo_levenshtein(row['original'], row['corrected']), axis=1)

    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)
    df['0_tokens'] = df['original'].apply(lambda x: len(sp(x)))
    df['1_tokens'] = df['corrected'].apply(lambda x: len(sp(x)))
    df['1_token/0_token'] = df['1_tokens'] / df['0_tokens']
    df['min_tokens'] = df[['0_tokens', '1_tokens']].min(axis=1)
    
    df = df[df['min_tokens'] > 0]
    df['log_tokens'] = df['min_tokens'].apply(lambda x: math.log(x, 20))
    df['ratio'] = df['jamo_levenshtein'] / df['min_tokens'] * df['log_tokens']
    
    df['0_len'] = df['original'].apply(lambda x: len(x))
    df['1_len'] = df['corrected'].apply(lambda x: len(x))
    df['len_ratio'] = df['1_len'] / df['0_len']

    return df

def apply_cond(df):
    token_ratio_cond = (df['1_token/0_token']<4) & (df['1_token/0_token']>0.25)
    len_ratio_cond = (df['len_ratio']<1.25) & (df['len_ratio']>0.5)
    min_token_cond = (df['min_tokens']>5)

    df = df[token_ratio_cond & len_ratio_cond & min_token_cond]
    df = df[(~df['corrected'].str.contains('or')) & (~df['corrected'].str.contains('good'))]
    df = df[df['original'].str.strip() != df['corrected'].str.strip()]
    df = df.drop_duplicates()

    return df

def main():
    args = parse_args()

    print('1. Load data')
    df = load_data(args.datafile)
    print('dataset shape:', df.shape, '\n')

    print('2. Drop NaN and rows whose original and corrected sentences are the same')
    df = df.dropna()
    df = df[df['original'].str.strip() != df['corrected'].str.strip()]
    df = df[(~df['original'].str.contains('[a-zA-Z]+')) & (~df['corrected'].str.contains('[a-zA-Z]+'))]
    print('dataset shape:', df.shape, '\n')

    print('3. Drop rows of other languages')
    df = drop_other_lang(df)
    print('dataset shape:', df.shape, '\n')

    print('4. Calculate variables and apply conditions')
    df = calc_vars(df)
    df = apply_cond(df)
    print('dataset shape:', df.shape, '\n')

    print('5. Save')
    df.to_csv(args.output, index=False)
    print('Done!')



if __name__ == '__main__':
    main()

