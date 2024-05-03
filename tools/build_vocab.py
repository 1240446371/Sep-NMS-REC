from tqdm import tqdm
import numpy as np
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from lib.refer import REFER
# remove re  word in sentence less than 2,and

GLOVE_WORD_NUM = 2196017
GLOVE_FILE = '/home_data/wj/ref_nms/data/glove.840B.300d/glove.840B.300d.txt'
VOCAB_THRESHOLD = 2
VOCAB_SAVE_PATH = '/home_data/wj/ref_nms/cache/std_vocab1_{}_{}.txt'
GLOVE_SAVE_PATH = '/home_data/wj/ref_nms/cache/std_glove1_{}_{}.npy'

# load sentence--get their all tokens, token times<2 is discarded, and sort them, then save them in voc file,no idx, not according sentence
def load_glove_feats():
    glove_dict = {}  # from word of <str> to vector of <generator>
    with open(GLOVE_FILE, 'r') as f:
        with tqdm(total=GLOVE_WORD_NUM, desc='Loading GloVe', ascii=True) as pbar:
            for line in f:
                #print("19line%s"%line)  # num
                tokens = line.split(' ')  # to string''
                #print("21tokens%s"%tokens)
                assert len(tokens) == 301
                word = tokens[0]
                #print("24word%s"%word)
                vec = list(map(lambda x: float(x), tokens[1:])) # transfer token to floats
                glove_dict[word] = vec
                #print("27vec%s"%vec)
                pbar.update(1)
    return glove_dict  # load an word and its features ,key is 'and',value is 0.14, 0.89,0.67.....


def build_vocabulary(dataset, split_by, glove_dict):
    # load refer image root
    refer = REFER('/home_data/wj/ref_nms/data/refer', dataset, split_by)

    # filter corpus by frequency and GloVe
    word_count = {}
    for ref in refer.Refs.values():
        for sent in ref['sentences']:
            for word in sent['tokens']:
                word_count[word] = word_count.get(word, 0) + 1  #count wordi times
    vocab, typo, rare = [], [], []
    for wd, n in word_count.items():  # rare word < 2 times
        if n < VOCAB_THRESHOLD:
            rare.append(wd)
        else:
            if wd in glove_dict:
                vocab.append(wd)
            else:
                typo.append(wd)  # wrong words
    assert len(vocab) + len(typo) + len(rare) == len(word_count)
    rare_count = sum([word_count[wd] for wd in rare])
    typo_count = sum([word_count[wd] for wd in typo])
    total_words = sum(word_count.values())
    print('number of good words: {}'.format(len(vocab)))
    print('number of rare words: {}/{} = {:.2f}%'.format(
        len(rare), len(word_count), len(rare)*100/len(word_count)))
    print('number of typo words: {}/{} = {:.2f}%'.format(
        len(typo), len(word_count), len(typo)*100/len(word_count)))
    print('number of UNKs in sentences: ({}+{})/{} = {:.2f}%'.format(
        rare_count, typo_count, total_words, (rare_count+typo_count)*100/total_words))

    # sort vocab and construct glove feats
    vocab = sorted(vocab)  # rank
    #print("69voc %s"%vocab[1])
    vocab_glove = []
    for wd in vocab:
        vocab_glove.append(glove_dict[wd])   # glove feats
    print("73voc_glove %s"%vocab_glove[1])
    vocab.insert(0, '<unk>')
    print("69voc %s"%vocab[1])
    vocab_glove.insert(0, [0.] * 300)
    print("76voc_glove %s"%vocab_glove[1])
    vocab_glove = np.array(vocab_glove, dtype=np.float32)

    # save vocab and glove feats
    vocab_save_path = VOCAB_SAVE_PATH.format(dataset, split_by) 
    glove_save_path = GLOVE_SAVE_PATH.format(dataset, split_by)
    print('saving vacob in {}'.format(vocab_save_path))
    with open(vocab_save_path, 'w') as f:
        for wd in vocab:
            f.write(wd + '\n')
    print('saving vocab glove in {}'.format(glove_save_path))
    np.save(glove_save_path, vocab_glove)


def main():
    print('building vocab...')
    glove_feats = load_glove_feats()
    for dataset, split_by in [('refcoco', 'unc'), ('refcoco+', 'unc'), ('refcocog', 'umd')]:
        print('building {}_{}...'.format(dataset, split_by))
        build_vocabulary(dataset, split_by, glove_feats)
    print()


main()
