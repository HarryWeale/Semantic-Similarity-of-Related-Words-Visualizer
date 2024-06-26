import glove
import numpy
import argparse

def compute_length(a):
    if a.ndim == 1:
        return numpy.linalg.norm(a)
    else:
        return numpy.linalg.norm(a, axis=1)

def cosine_similarity(array1, array2):
    dot_product = numpy.dot(array2, array1)
    length1 = compute_length(array1)
    length2 = compute_length(array2)
    finalSimil = numpy.divide(dot_product, length2*length1)
    return finalSimil

def closest_vectors(v, words, array, n):
    newList = []
    for item in zip(cosine_similarity(v, array), words):
        newList.append(item)
    newList.sort(key=lambda tup: tup[0], reverse=True)
    return newList[:n]

def main(args):
    a = numpy.array([1,2,3,4])
    b = numpy.array([[1,2,3,4],[5,6,7,8]])
    a1 = numpy.array([1,2,3,4,5])
    b1 = numpy.array([0,1,0,1,0])
    c1 = numpy.array([[0,1,0,1,0],[1,0,1,0,1]])

    vectors, word_list = glove.load_glove_vectors(args.npyFILE)

    # print(word_list[:10])
    # print(word_list.index('cat'))
    dog_vec = glove.get_vec('dog', word_list, vectors)
    # print(compute_length(dog_vec))
    cat_vec = glove.get_vec('cat', word_list, vectors)
    # print(cosine_similarity(dog_vec, cat_vec))
    
    if args.word:
        words = [args.word]
    elif args.file:
        words = [x.strip() for x in args.file]
    else: return

    for word in words:
        print("The {} closest words to {} are:".format(args.num, word))
        word_vector = glove.get_vec(word, word_list, vectors)
        closest = closest_vectors(word_vector, word_list, vectors, args.num)
        print(closest)
        for similarity, similar_word in closest:
            print("    * {} (similarity {})".format(similar_word, similarity))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the n closest words to " +\
                      "a given word (if specified) or to all of the " +\
                       "words in a text file (if specified). If " +\
                       "neither is specified, compute nothing.")
    parser.add_argument("--word", "-w", metavar="WORD", help="a single word")
    parser.add_argument("--file", "-f", metavar="FILE", type=argparse.FileType('r'),
                        help="a text file with one word per line.")
    parser.add_argument("--num", "-n", type=int, default=5,
                        help="find the top n most similar words")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('rb'),
                        help='an .npy file to read the saved numpy data from')

    args = parser.parse_args()
    main(args)