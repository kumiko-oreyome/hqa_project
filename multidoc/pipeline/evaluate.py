from . import config
from multidoc.eval import  DureaderMultiDocMrcEvaluator
from multidoc.core.op import create_reader,create_judger,create_ranker


def evaluate_dureader(paths,ranker,reader,judger):
    ranker = create_ranker(ranker)
    reader = create_reader(reader)
    judger = create_judger(judger)
    print('evaulate pipeline of dureader %s'%(str(paths)))
    e = DureaderMultiDocMrcEvaluator('dureader')
    res = e.evaluate_multidoc_mrc_on_path(paths,ranker,reader,judger)
    print(res)
    
    
    
if __name__ == '__main__':
    #dureader_dev_paths = ['./data/search.dev.json','./data/zhidao.dev.json']
    debug_file = './data/demo/devset/search.dev.json'
    #evaluate_dureader(dureader_dev_paths,config.word_match_ranker,config.bert_reader_old,config.default_judger)
    evaluate_dureader(debug_file,config.bert_ranker_old,config.bert_reader_old,config.default_judger)