# Compute the overlapping score for a pair of sentences

import pandas as pd
import sys
from customeclass import QuestionPair

# pd.set_option('expand_frame_repr', False)
# pd.set_option('max_rows', None)
# pd.set_option('max_colwidth', 150)

# get first argument as a csv file
argument = sys.argv[1]
csv_data = pd.read_csv(argument, header = None,
                       names = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])
# csv_data = csv_data[0:10]

# Outputs the overlapping score for all question pairs
for index, row in csv_data.iterrows():
    questPair = QuestionPair()
    questPair.setQuestions(row["question1"], row["question2"])
    print(questPair.overlappingScore())

# Creates the "overlappingScore" column and fills it with data
csv_data["overlappingScore"] = 0.0
for index, row in csv_data.iterrows():
    questPair = QuestionPair()
    questPair.setQuestions(row["question1"], row["question2"])
    csv_data.loc[index, "overlappingScore"] = questPair.overlappingScore()

# print("The maximum overlapping score is " + str(csv_data["overlappingScore"].max()))
# print("The minimum overlapping score is " + str(csv_data["overlappingScore"].min()))
# print("The median overlapping score is " + str(csv_data["overlappingScore"].median()))
#
# sorted_csv_data = csv_data.sort_values(by = ["overlappingScore"], ascending = False)
#
# print(sorted_csv_data.head())
# print(sorted_csv_data.tail())
