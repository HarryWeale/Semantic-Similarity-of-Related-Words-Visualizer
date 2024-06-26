import glove
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from similarity import *
import argparse
from itertools import chain
import numpy as np
from sklearn.decomposition import PCA

def perform_pca(array, n_components):
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(array)   
    return pc

def read_relations(fp):
    allPairs = []
    firstLine = 0
    for line in fp:
        if firstLine == 0:
            firstLine += 1
            continue
        pair = line.split()
        allPairs.append((pair[0], pair[1]))
    return allPairs

def plot_relations(pca_first, pca_second, pca_relations, filename='drinks.png'):
    fig = plt.figure(figsize = (8,8)) 
    ax = fig.add_subplot(1,1,1)

    ax.scatter(pca_first[:,0], pca_first[:,1], c='r', s=50)
    ax.scatter(pca_second[:,0], pca_second[:,1], c='b', s=50)

    for i in range(len(pca_first)):
        (x,y) = pca_first[i]
        plt.annotate(pca_relations[i][0], xy=(x,y), color="black")
        (x,y) = pca_second[i]
        plt.annotate(pca_relations[i][1], xy=(x,y), color="black")

    for i in range(len(pca_first)):
        (x1,y1) = pca_first[i]
        (x2,y2) = pca_second[i]
        ax.plot((x1, x2), (y1, y2), linewidth=1, color="lightgray")

    plt.savefig(filename)
    return

def extract_words(vectors, word_list, relations):
    numRows = 0
    numCols = 0
    correctRelations = []
    for relation in relations:
        # Check to make sure both elements in the tuple are valid.
        if relation[0] in word_list and relation[1] in word_list:
            numRows += 1
            firstWordIndex = word_list.index(relation[0])
            numCols = len(vectors[firstWordIndex])
            correctRelations.append(relation)

    firstWordVectors = np.zeros((numRows, numCols))
    secondWordVectors = np.zeros((numRows, numCols))

    wordArraysIndex = 0
    for correctRelation in correctRelations:
        firstWordVectors[wordArraysIndex] = vectors[word_list.index(correctRelation[0])]
        secondWordVectors[wordArraysIndex] = vectors[word_list.index(correctRelation[1])]
        wordArraysIndex += 1

    array = numpy.vstack((firstWordVectors, secondWordVectors))
    pca_vectors = perform_pca(array, 2)

    pca_first = pca_vectors[:len(correctRelations)]
    pca_second = pca_vectors[len(correctRelations):]

    plot_relations(pca_first, pca_second, correctRelations)
    return (firstWordVectors, secondWordVectors, correctRelations)

def main(args):
    vectors, word_list = glove.load_glove_vectors(args.npyFILE)
    relations = read_relations(args.relationsFILE)
    # relations = [('brother', 'sister'), ('nephew', 'niece'), ('werewolf', 'werewoman')]
    extract_words(vectors, word_list, relations)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')
    parser.add_argument("relationsFILE",
                        type=argparse.FileType('r'),
                        help='a file containing pairs of relations')
    parser.add_argument("--plot", "-p", default="plot.png", help="Name of file to write plot to.")
    args = parser.parse_args()
    main(args)