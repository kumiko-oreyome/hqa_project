from . import DureaderEvaluator


def test_metrics():
    print('test_match_based_similarity')
    s = ["aa","b","cc","ff"]
    ref = ["b","aaa","d"]
    s = "aa b cc ff"
    ref = "b aaa d"
    

def test_dureader_evaluator():
    DureaderEvaluator().evaluate_reader_on_path('./data/demo/trainset/search.train.json',None)


if __name__ == '__main__':
    test_dureader_evaluator()