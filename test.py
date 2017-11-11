import numpy as np
import os


def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print (full_filename)


def search2(dirname):
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            print(path, dir, filename)

# search('data/omniglot/Korean/character02')
search2('./data/omniglot')