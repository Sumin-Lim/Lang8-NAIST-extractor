'''
Description: This file preprocesses and make Lang-8 dataset.

Usage: python preprocessing.py -d [file path] -o [file path]
-d: pickled result file path after running extract_err-cor-pair.py
-o: file path to save the result of this script
'''
import pickle as pkl
from tqdm import tqdm 
import pandas as pd
import re
import argparse

import math
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from Levenshtein import distance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', required=True, nargs=1, type=str)
    parser.add_argument('-o', '--output', required=True, nargs=1, type=str)
    args = parser.parse_args()
    return args

def load_data(file_path):
    dataset = [] 
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
        for line in tqdm(data):
            dataset.append(line.strip().split('\t'))
    return dataset

def is_hangul(x):
    # detect the ratio of hangul 
    han_count = ' '.join(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', x))
    return len(han_count) / len(x)

def is_chinese(x):
    # detect chinese
    return len(re.findall(u'[\u4e00-\u9FFF]', x)) > 0

def split_space(x):
    # detect one word, to drop them 
    return len(x.split()) == 1

def drop_non_hangul(df):
    # drop non korean sentences and drop rows with the ratio of korean under 70%
    df = df.dropna()
    df = df.copy()

    df['0_hangul'] = df[0].apply(lambda x: is_hangul(x))
    df['1_hangul'] = df[1].apply(lambda x: is_hangul(x))
    df = df.loc[(df['0_hangul'] > 0.7) & (df['1_hangul'] > 0.7)][[0, 1]]
    return df

def drop_chinese(df):
    # drop sentences containing chinese characters
    df = df.dropna()
    df = df.copy()

    df['0_chinese'] = df[0].apply(lambda x: is_chinese(x))
    df['1_chinese'] = df[1].apply(lambda x: is_chinese(x))
    df = df.loc[(df['0_chinese']==False) & (df['1_chinese']==False)][[0, 1]]
    return df

def str_replace(df):
    # drop rows with values of <-, ->, /
    # replace strings in parentheses to null string
    df['0_replace'] = df[0].str.replace(r'\(.*\)', '')
    df['1_replace'] = df[1].str.replace(r'\(.*\)', '')

    df = df[~df['0_replace'].str.contains('<-')]
    df = df[~df['0_replace'].str.contains('->')]
    df = df[~df['0_replace'].str.contains('/')]

    df = df[~df['1_replace'].str.contains('<-')]
    df = df[~df['1_replace'].str.contains('->')]
    df = df[~df['1_replace'].str.contains('/')]
    return df

def drop_one_word(df):
    # drop one word sentences
    df['0_split'] = df['0_replace'].apply(lambda x: split_space(x))
    df['1_split'] = df['1_replace'].apply(lambda x: split_space(x))
    df = df.loc[(df['0_split']==False) & (df['1_split']==False)][['0_replace', '1_replace']]
    return df


def get_ratio(*x):
    # calculate ratio
    distance = x[0]
    min_words = x[1]
    return distance / float(min_words) * math.log(min_words, 20)

def make_ratio(df):
    # make ratio column in dataframe
    df['is_same'] = df['original'].str.strip() == df['corrected'].str.strip()
    df = df[df['is_same']==False][['original', 'corrected']]

    tok_path = get_tokenizer()
    sp = SentencepieceTokenizer(tok_path)

    def get_min_words(*x):
        # get the minimum number of tokens in two sentences, which is used in calculating ratio
        old_tokens = sp(x[0])
        new_tokens = sp(x[1])
        return min(len(old_tokens), len(new_tokens))

    df['min_words'] = df[['original', 'corrected']].apply(lambda x: get_min_words(*x), axis=1)
    df['distance'] = df[['original', 'corrected']].apply(lambda x: distance(*x), axis=1)
    df = df[df['min_words'] != 0]
    df['ratio'] = df[['distance', 'min_words']].apply(lambda x: get_ratio(*x), axis=1)
    return df

def main():
    args = parse_args()

    print('1. Load data')
    df = pd.DataFrame(load_data(args.datafile[0]))

    print('2. Drop rows with the ratio of Korean under 70% and rows containing Chinese')
    df_drop_hangul = drop_non_hangul(df)
    df_drop_chinese = drop_chinese(df_drop_hangul)

    print('3. Drop rows with special characters and rows with one word')
    df_replace = str_replace(df_drop_chinese)
    df_drop = drop_one_word(df_replace)

    df = df_drop.rename(columns={'0_replace':'original',
                                 '1_replace':'corrected'})
    print('4. Make edit ratio')
    df = make_ratio(df)
    df = df[df['ratio'] < 1]

    df = df.dropna()
    df = df.drop_duplicates()
    print('Data shape:', df.shape)
    
    print('5. Save data')
    df_sorted = df.sort_values(by='ratio', ascending=False)
    df_sorted.to_csv(args.output[0], index=False, encoding='utf-8')


if __name__ == '__main__':
    main()

