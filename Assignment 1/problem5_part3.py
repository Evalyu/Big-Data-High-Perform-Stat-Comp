import pandas as pd
import sys
import numpy as np
from customeclass import QuestionPair

# pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', None)
# pd.set_option('max_colwidth', 150)

# get first argument as a csv file
argument = sys.argv[1]

csv_data = pd.read_csv(argument, header = None,
                       names = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])

# Creates the "overlappingScore" column and fills it with data
threshold = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
accuracy = {}

for thr in threshold:
    accuratePredictions = 0
    for index, row in csv_data.iterrows():
        questPair = QuestionPair()
        questPair.setQuestions(row["question1"], row["question2"])
        thresholdOverlap = np.sign(questPair.overlappingScore() - threshold)
        # If my prediction says the pair of questions are similar or 50/50 and is_duplicate == 1, then +1
        if row["is_duplicate"] == 1 and (thresholdOverlap == 1 or thresholdOverlap == 0):
            accuratePredictions += 1
        # If my prediction says the pair of questions are not similar and is_duplicate == 0, then +1
        if row["is_duplicate"] == 0 and thresholdOverlap == -1:
            accuratePredictions += 1

    total_pairs = len(csv_data)
    test_accuracy = round(float(accuratePredictions) / total_pairs, 4)
    accuracy[thr] = test_accuracy

print(max(accuracy, key=accuracy.get)) # thr = 0.5

# Output
# {0.1: 0.4348, 0.2: 0.4989, 0.5: 0.6639, 1.0: 0.6341, 0.7: 0.6464, 0.65: 0.653, 0.3: 0.5739, 0.8: 0.6332, 0.55: 0.6647,
# 0.45: 0.6558, 0.4: 0.635, 0.6: 0.6601}
# 0.55
