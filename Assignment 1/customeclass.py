from problem1 import preprocess

# The dataset for two questions
class QuestionPair:
    question1 = None # type: list[str]
    question2 = None # type: list[str]

    def setQuestions(self, q1, q2):
        """
        Preprocess the questions, then stores them as a list of words
        
        :param str q1: 
        :param str q2: 
        :return: 
        """
        self.question1 = preprocess(q1).split()
        self.question2 = preprocess(q2).split()

    def overlappingScore(self):
        """
        Checks each word in question1 and question 2 to see if it is in the other question.
        Add 1 to the numerator if the word is in the other question.
        Denominator is the total number of words in both question1 and question2
        
        :return:
        """
        numerator = 0

        for word in self.question1:
            if word in self.question2:
                numerator += 1

        for word in self.question2:
            if word in self.question1:
                numerator += 1

        denominator = len(self.question1) + len(self.question2)

        if denominator == 0:
            score = 0
        else:
            score = numerator / denominator

        return score

    def removeWords(self, words):
        """
        This method removes words from a given list in the two questions
        
        :param list[str] words: 
        :return: 
        """

        q1_filter = []
        q2_filter = []

        # If word is not in words, add it to the filter list
        for word in self.question1:
            if (word in words) == False:
                q1_filter.append(word)

        for word in self.question2:
            if (word in words) == False:
                q2_filter.append(word)

        self.question1 = q1_filter
        self.question2 = q2_filter

    def __str__(self):
        return "Q1: " + str(self.question1) + " Q2: " + str(self.question2)
