from glove import *
from similarity import *
from visualize import *
import argparse
import numpy as np
import random
import sys

def average_difference(first_vectors, second_vectors):
    vec_list = []
    for line in range(len(first_vectors)):
        vec_list.append(first_vectors[line] - second_vectors[line])
    vec_average = sum(vec_list) / len(vec_list)
    return vec_average

def do_experiment(args):
    vectors, word_list = load_glove_vectors(args.npyFILE)
    relations = read_relations(args.relationsFILE)  

    random.shuffle(relations)
    eighty = int(0.8*len(relations))
    training_relations = relations[:eighty]
    test_relations = relations[eighty:]

    first_vectors, second_vectors, filtered_relations = extract_words(vectors, word_list, training_relations)
    first_vectors_test, second_vectors_test, filtered_relations_test = extract_words(vectors, word_list, test_relations)  

    vecToAdd = average_difference(first_vectors, second_vectors)

    topTenCount = 0
    firstCount = 0
    everythingCount = len(second_vectors_test)
    for second in range(len(second_vectors_test)):
        result = closest_vectors(second_vectors_test[second] + vecToAdd, word_list, vectors, 101)
        result = result[1:]
        firstWord = filtered_relations_test[second][0]
        for word in range(10):
            if result[word][1] == firstWord:
                topTenCount += 1    
                if word == 0:
                    firstCount += 1
    topTenRatio = topTenCount/everythingCount
    # print(topTenRatio, "top ten")
    firstRatio = firstCount/everythingCount
    # print(firstRatio, "first")
    return

def main(args):
    do_experiment(args)
    
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
    args = parser.parse_args()
    main(args)