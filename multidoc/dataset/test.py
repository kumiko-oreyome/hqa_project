from .dureader import DureaderExample,load_dataset
def test_load_data():    
    check_answer_doc = lambda x:None if DureaderExample(x).illegal_answer_doc() else DureaderExample(x)
    select_gold_paragraph = lambda example:DureaderExample(example.copy_subset_documents([])).update('documents',\
        [{'paragraphs':[''.join(example.get_gold_paragraph(['segmented_paragraphs'])['segmented_paragraphs'])]}]).update('span',example.get_gold_span())
    flatten_func  = lambda example:example.flatten(['question_id','question','span','fake_answers'],[])
    preprocess_funcs = [ check_answer_doc,select_gold_paragraph,flatten_func]
    examples = load_dataset('./data/demo/devset/search.dev.json',preprocess_funcs)
    for example in examples:
        assert example['fake_answers'][0] == example['passage'][example["span"][0]:example["span"][1]+1]
    
test_load_data()