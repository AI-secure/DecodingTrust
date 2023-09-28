import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--out-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--total', type=int, default=-1,
                    help='Total number of data')
parser.add_argument('--workers', type=int, default=1,
                   help='Number of worker processes to launch')



import re

def split_sentence(input_string):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(input_string)
    tot = len(sentences)
    if len(sentences) == 1:
        return "", input_string
    else:
        start = tot // 2
        return " ".join(sentences[:start]), " ".join(sentences[start:])

def main():
    args = parser.parse_args()

    path = args.data_path
    out = args.out_path if args.out_path else path + '.split'
    print(out)

    with open(path) as f:
        lines = f.readlines()
    import json

    data = [json.loads(x) for x in lines]

    continuations = []

    for x in data:
        continuation = []
        for y in x['response']['choices']:
            continuation.append(y['message']['content'])
        continuations.append(continuation)

    splitted_corpus = []

    for i, cont in enumerate(continuations):
        for x in cont:
            a, b = split_sentence(x)
            if b:
                splitted_corpus.append((b, i))

    print(len(splitted_corpus))

    with open(out, 'w') as f:
        for x in splitted_corpus:
            for _ in range(1):
                f.write(json.dumps({"text": x[0], "idx": x[1]}) + '\n')


if __name__ == '__main__':
    main()

