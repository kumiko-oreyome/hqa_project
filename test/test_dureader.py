from multidoc.util import jsonl_reader
from multidoc.dataset.dureader import DureaderExample
def test_char_preprocessing():
    path = './data/search.train.json'
    for example in jsonl_reader(path):
        ex  = DureaderExample( example )
        if ex.illegal_answer_doc() or 'fake_answers' not in example or len(example['fake_answers'])==0:
            continue
        fake_answer = example['fake_answers'][0]
        aaa = ex.charspan_preprocessing('gold_span')
        answer_doc = ex.get_answer_doc()[0]
        p = answer_doc['paragraphs'][answer_doc["most_related_para"]]
        start_pos,end_pos = aaa.sample_obj['gold_span']
        try:
            assert fake_answer == aaa.sample_obj['documents'][0]['paragraphs'][0][ start_pos:end_pos+1]
        except:
            import pdb;pdb.set_trace()
            

def test_select_fields():
    doc = {'question':'c8763?','answer_docs':[1],'documents':[{'paragraphs':['aaa','bbb','ccc'],'segmented_paragraphs':[['a','aa'],['bbb'],['c','c','c']],'title':'10 secs','most_related_para':1},{'paragraphs':['motohayaku','switch !','start burst stream'],'segmented_paragraphs':[['moto','haya','ku'],['switch','!'],['start','burst','stream']],'title':'owataka?','most_related_para':2}]}
    
    aaa = DureaderExample(doc).select_fields(['question','answer_docs','documents'],['most_related_para'],['paragraphs','segmented_paragraphs'])
    aaa = DureaderExample(aaa)
    print(aaa.select_by_indexs('all','gold_paragraph',['paragraphs','segmented_paragraphs']))
    #print(aaa.select_by_indexs('answer_doc','gold_paragraph',['paragraphs','segmented_paragraphs']))

    
            
test_select_fields()
#test_char_preprocessing()
