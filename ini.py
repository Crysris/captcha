import os
import shutil

ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


def move(src='test', dst='train_data'):
    files = os.listdir(src)
    for file in files:
        if len(file) == 10:
            path = os.path.join(src, file)
            shutil.move(path, dst)


def rlist(src):
    files = os.listdir(src)
    for file in files:
        path = os.path.join(src, file)
        if os.path.isdir(path):
            rlist(path)
        else:
            print(path)


def clean(src='train_data', dst='me_learn2'):
    files = os.listdir(src)
    for file in files:
        name = file.split('.')[0]
        path = os.path.join(src, file)
        for c in ALPHABET:
            if c in name:
                shutil.move(path, dst)
                break


move()
clean()
