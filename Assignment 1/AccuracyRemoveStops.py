# Remove stopwords that appear more than 1000 times in training.csv

import pandas as pd
import sys
import numpy as np
from collections import Counter
from problem1 import preprocess
from customeclass import QuestionPair

# pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', None)
# pd.set_option('max_colwidth', 150)

# Get first argument as a csv file
argument = sys.argv[1]
threshold = float(sys.argv[2])

csv_data = pd.read_csv(argument, header = None,
                       names = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])

# Puts all the words from all the questions into on giant list
# This list will be used to count the most common words
entire_word_list = str()

entire_word_list = preprocess(entire_word_list)

# Counts the number of occurrences of each word and puts it into word_count
word_count = Counter(entire_word_list.split()) # type: dict[str, int]

# Filters by words that appear over 1000 times and sorts them by their appearance amount
stop_words = list(word for word, count in word_count.items() if count > 1000)

# Compute the training accuracy
# Creates the "overlappingScore" column and fills it with data
accuratePredictions = 0

for index, row in csv_data.iterrows():
    questPair = QuestionPair()
    questPair.setQuestions(row["question1"], row["question2"])
    questPair.removeWords(stop_words)
    overlappingScoreThresh = questPair.overlappingScore() - threshold
    thresholdOverlap = np.sign(overlappingScoreThresh)
    # If my prediction says the pair of questions are similar or 50/50 and is_duplicate == 1, then +1
    if row["is_duplicate"] == 1 and (thresholdOverlap == 1 or thresholdOverlap == 0):
        accuratePredictions += 1
    # If my prediction says the pair of questions are not similar and is_duplicate == 0, then +1
    if row["is_duplicate"] == 0 and thresholdOverlap == -1:
        accuratePredictions += 1
    # p.incrementOne()

total_pairs = len(csv_data)
test_accuracy = round(float(accuratePredictions) / total_pairs, 4)
print(test_accuracy)

# training.csv
# Threshold: 0.1, Accuracy: 0.5013
# Threshold: 0.2, Accuracy: 0.5182
# Threshold: 0.3, Accuracy: 0.563
# Threshold: 0.4, Accuracy: 0.5915
# Threshold: 0.45, Accuracy: 0.6558
# Threshold: 0.5, Accuracy: 0.6194
# Threshold: 0.55, Accuracy: 0.6647
# Threshold: 0.6, Accuracy: 0.6453
# Threshold: 0.65, Accuracy: 0.653
# Threshold: 0.7, Accuracy: 0.6464
# Threshold: 0.8, Accuracy: 0.6332
# Threshold: 1.0, Accuracy: 0.6341
# Best = 0.55

# validation.csv
# Threshold: 0.55, Accuracy: 0.6647