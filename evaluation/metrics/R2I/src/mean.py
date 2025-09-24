import os
import sys
import pandas as pd
import argparse

def compute_mean_rank(experiment: str, targets: list[str]):
    maxr = len(targets)
    assert maxr > 1, "At least two targets are required"

    data = []
    bin_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval', experiment, 'bin')
    os.chdir(bin_dir)

    for filename in os.listdir():
        b = pd.read_csv(filename, header=0, delimiter=',')
        data.append(b)

    data = pd.concat(data, axis=0)
    index = data.filter(['decompiler', 'binary'])
    data = index.join(data.groupby(by=['decompiler', 'binary'])['r2i'].mean(), on=['decompiler', 'binary'])
    data = data.drop_duplicates()

    with open('../mean.csv', 'w') as f:
        data.to_csv(f, index=False)

    ranks = []
    for i in range(len(data) // maxr):
        i *= maxr
        ranks.extend(list(data.iloc[i:i+maxr, 2].rank(method='min', ascending=False)))
    data['rank'] = ranks

    ranked = {decompiler: [0] * maxr for decompiler in targets}
    for dv in data.values:
        ranked[dv[0]][int(dv[3]) - 1] += 1
    ranked = pd.DataFrame(ranked)

    with open('../rom.csv', 'w') as f:
        ranked.to_csv(f, index=False)

    # print(ranked)

    os.remove('../rom.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('-t', '--targets', nargs='+', required=True,
                            help='specify targets to relatively aggregate')
    args = parser.parse_args()

    compute_mean_rank(args.experiment, args.targets)