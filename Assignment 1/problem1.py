#problem 1: processing

def preprocess(str_in):
    """
    changes str_in to lowercase and replaces ? , ! . ( ) ' " : with " " and removes all "-"
    
    :param str str1_out:
    :return:
    """

    str1_out = str(str_in).lower()

    punct_list = '?,!.()\'":'

    for p in punct_list:
        str1_out = str1_out.replace(p, " ")

    str1_out = str1_out.replace("-", "")

    return str1_out