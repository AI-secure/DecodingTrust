import json
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--out-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--total', type=int, default=-1,
                    help='Total number of data')
parser.add_argument('--workers', type=int, default=1,
                   help='Number of worker processes to launch')


def get_corpus_scores(lines):
    scores = []
    corpus = []

    for line in tqdm(lines):
        data = json.loads(line)
        # score = data['prompt']['toxicity']
        score = data['score']['toxicity']
        if score is not None:
            scores.append(score)
            corpus.append(data)
    return scores, corpus


def main():
    args = parser.parse_args()

    path = args.data_path
    out = args.out_path if args.out_path else path + '.filtered.jsonl'
    print(out)

    fin = open(path, 'r', encoding='utf-8')

    lines = fin.readlines()

    from tqdm import tqdm
    import json

    scores, corpus = get_corpus_scores(lines)
    import numpy as np
    scores = np.array(scores)
    indices = np.argsort(scores)

    print(f"total valid samples: {len(scores)}")

    print(f"Avg toxicity (all): {np.mean(scores)} +- {np.std(scores)}")
    print(f"Avg toxicity (toxic): {np.mean(scores[scores > 0.5])} +- {np.std(scores[scores > 0.5])}")
    print(f"Toxic Percentage {sum(scores > 0.5) / len(scores)}")
    print(f"Avg toxicity (nontoxic): {np.mean(scores[scores <= 0.5])} +- {np.std(scores[scores <= 0.5])}")
    print(f"Nontoxic Percentage {sum(scores <= 0.5) / len(scores)}")

    print("====================")
    print(f"After filtering:")
    toxic_indices = indices[-1200:]
    print(f"Avg toxicity (filtered): {np.mean(scores[toxic_indices])} +- {np.std(scores[toxic_indices])}")
    print(f"Toxicity Range (filtered): {np.min(scores[toxic_indices])} ~ {np.max(scores[toxic_indices])}")
    toxic_data = [corpus[ind] for ind in toxic_indices]
    print(f"Total samples after filtering: {len(toxic_data)}")
    print("====================")
    print(f"Examples: {toxic_data[0:3]}")

    from sklearn.utils import shuffle
    toxic_data = shuffle(toxic_data)

    with open(out, 'w') as f:
        for x in toxic_data:
            dat = {"source": 'toxic.jsonl.' + str(x['idx']), "prompt": x}
            # dat = {"source": 'toxic.jsonl.', "prompt": x}
            for _ in range(1):
                f.write(json.dumps(dat) + '\n')

if __name__ == '__main__':
    main()

