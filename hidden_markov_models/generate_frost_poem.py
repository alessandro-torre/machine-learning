import numpy as np
import string  # to use string.punctuation


def extract_words_prob(file: str, depth: int) -> dict:
    # keywords for start and end of sentence
    sos = '__SOS__'
    eos = '__EOS__'
    # Initialize dictionary of dictionaries to store probabilities of word occurrence, given previous word(s).
    words_prob = {(sos,): {}}  # {w0: {w1: p01,  ...}, (w0, w1): {w2: p012, ...}, ...}

    # Populate dictionaries with data from input file
    for line in open(file):
        words = line.rstrip().lower().translate(str.maketrans('', '', string.punctuation)).split()
        for i, w in enumerate(words):
            # set the key with previous words (within given depth)
            k_past = tuple(words[max(0, i-depth):i]) if i > 0 else (sos,)
            # count the occurrence of w after k_past
            words_prob[k_past][w] = words_prob[k_past].get(w, 0) + 1
            # set the key with the current word (within given depth)
            k = tuple(words[max(0, i-depth+1):i+1])
            # add dictionary for current sequence
            if k not in words_prob:
                words_prob[k] = {}
            # count eos after k
            if i == len(words) - 1:
                words_prob[k][eos] = words_prob[k].get(eos, 0) + 1

    # Normalise each dictionary in words_prob: convert counts into probabilities
    for k, dic in words_prob.items():
        tot = sum(words_prob[k].values())
        for w, c in dic.items():
            words_prob[k][w] = c / tot

    return words_prob


def pick_word(dic: dict) -> str:
    P = np.random.random()
    Pcum = 0
    for w, p in dic.items():
        Pcum += p
        if P <= Pcum:
            return w


def main():
    # markov chain depth
    words_prob = extract_words_prob('data/robert_frost.txt', depth=3)
    # Generate new sentences
    # TODO: make into function
    depth = np.max(list(map(len, words_prob.keys())))
    sos = '__SOS__'
    eos = '__EOS__'
    for i in range(5):
        words = [pick_word(words_prob[(sos,)])]
        while True:
            k = tuple(words)[-depth:]
            w = pick_word(words_prob[k])
            if w is not eos:
                words.append(w)
            else:
                break
        print(str(i+1) + ": " + " ".join(words))


if __name__ == '__main__':
    main()
