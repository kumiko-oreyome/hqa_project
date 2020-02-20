import torch
from multidoc.dataset.dureader import load_dureader_examples,flatten_examples
from  multidoc.util.bert import BertReader
from  multidoc.util import get_default_device,JsonConfig
from  multidoc.eval import DureaderMultiDocMrcEvaluator


def demo():
    #config_path = './models/reader/bert_reader/dureader/debug/config.json'
    #config_path = './models/reader/bert_reader/drcd/debug/config.json'
    config_path = './models/reader/bert_reader/dureader/config.json'
    #config_path = './models/reader/bert_reader/drcd/config.json'
    config = JsonConfig(config_path)
    reader = BertReader.from_config(config)
    examples = load_dureader_examples(config.get_values('dev_paths'),'answer_doc')
    examples = flatten_examples(examples,['question','question_id','answers'],[],[])
    predictions = reader.evaluate_on_records(examples)
    for pred in predictions[0:20]:
        print('Question: %s'%(pred['question']))
        print('Passage:')
        print(pred['passage'])
        print('Prediction:')
        print(pred['span'])
        print('Answer:')
        print(pred['answers'][0])
        print('- - - -'*10)

def evaluate():
    #config_path = './models/reader/bert_reader/drcd/debug/config.json'
    #config_path = './models/reader/bert_reader/dureader/debug/config.json'
    #config_path = './models/reader/bert_reader/dureader/config.json'
    #config_path = './models/reader/bert_reader/old/config.json'
    config_path = './models/reader/bert_reader/origin_ver/config.json'
    config = JsonConfig(config_path)
    reader = BertReader.from_config(config)
    result = DureaderMultiDocMrcEvaluator(config.get_values('metric_type')).evaluate_reader_on_path(config.get_values('dev_paths'),'gold_paragraph',reader)
    print(result)
    
if __name__ == '__main__':
    evaluate()
    #demo()