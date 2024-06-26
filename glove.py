import argparse
import numpy

def load_text_vectors(fp):
    numRows = 0
    numCols = 0
    while fp.readline():
        currLine = fp.readline()
        splitLine = currLine.split(" ")
        numRows += 1
        numCols = len(splitLine) - 1
    vectors = numpy.zeros((numRows, numCols))
    fp.seek(0)

    listOfWords = []
    index = 0
    while numRows != 0:
        currLine = fp.readline()
        numRows = numRows - 1
        eachWord = currLine.split(" ")
        listOfWords.append(eachWord[0])
        vector = eachWord[1:]
        vector[-1] = vector[-1].replace("\n", '')
        floatsArray = []
        for number in vector:
            floatsArray.append(float(number))
        vectors[index] = floatsArray
        index += 1
    return (listOfWords, vectors) 

def save_glove_vectors(word_list, vectors, fp):
    numpy.save(fp, vectors)
    numpy.save(fp, word_list)
    fp.close()

def load_glove_vectors(fp):
    array = numpy.load(fp, allow_pickle=True)
    words = list(numpy.load(fp, allow_pickle=True))
    return (array, words)

def get_vec(word, word_list, vectors):
    desiredIndex = word_list.index(word)
    return vectors[desiredIndex]

def main(args):
    words, vectors = load_text_vectors(args.GloVeFILE)
    save_glove_vectors(words, vectors, args.npyFILE)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("GloVeFILE",
                        type=argparse.FileType('r'),
                        help="a GloVe text file to read from")
    parser.add_argument("npyFILE",
                        type=argparse.FileType('wb'),
                        help='an .npy file to write the saved numpy data to')

    args = parser.parse_args()
    main(args)
