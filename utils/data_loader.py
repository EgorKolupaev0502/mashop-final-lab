import pandas as pd


def load_file(file):
    data = pd.read_csv(file)
    sentences = data['sentence']
    aspects = data.iloc[:, 3:]
    return sentences, aspects


def load_files(*files):
    return list(map(lambda x: pd.concat(x, ignore_index=True),
                    zip(*[load_file(file) for file in files])))
