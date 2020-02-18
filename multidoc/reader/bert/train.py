import torch
from tqdm import tqdm
from torch.optim import SGD
from multidoc.dataset.dureader import load_dataset,DureaderExample
from  multidoc.util.bert import create_bert_model,BertDataset,BertInputConverter,BertReader
from  multidoc.util.bert.optimizer import get_bert_optimizer
from  multidoc.util import MetricTracer,get_default_device,JsonConfig
from  multidoc.core.op import LinearDecoder
from  multidoc.eval import DureaderMultiDocMrcEvaluator


def gold_span_preprocessing_dureader(source,max_q_len,max_seq_length):
    if (len(source['answer_spans']) == 0):
        return None
    if source['answers'] == []:
        return None
    if (source['match_scores'][0] < 0.8):
        return None
    if (source['answer_spans'][0][1] > max_seq_length):
        return None

    docs_index = source['answer_docs'][0]

    start_id = source['answer_spans'][0][0]
    end_id = source['answer_spans'][0][1] + 1          ## !!!!!
    question_type = source['question_type']

    passages = []
    try:
        answer_passage_idx = source['documents'][docs_index]['most_related_para']
    except:
        return None

    doc_tokens = source['documents'][docs_index]['segmented_paragraphs'][answer_passage_idx]
    # 去掉段落內的標題(大部分的段落一開始都會重複標題和一個句號)
    ques_len = len(source['documents'][docs_index]['segmented_title']) + 1
    doc_tokens = doc_tokens[ques_len:]
    start_id , end_id = start_id -  ques_len, end_id - ques_len

    if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
        return None

    new_doc_tokens = ""
    for idx, token in enumerate(doc_tokens):
        if idx == start_id:
            new_start_id = len(new_doc_tokens)
            break
        new_doc_tokens = new_doc_tokens + token

    new_doc_tokens = "".join(doc_tokens)

    new_end_id = new_start_id + len(source['fake_answers'][0])            
    if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id:new_end_id]):
        return None
    new_end_id = new_end_id - 1
    # check this example will exceed max_seq_len after convert
    q_len = min(max_q_len,len(source['question'].strip()))
    if q_len+3+new_end_id>max_seq_length:
        return None
    
    example = {
            "question_id":source['question_id'],
            "question":source['question'].strip(),
            "passage":new_doc_tokens.strip(),
            "gold_span": (new_start_id,new_end_id)}
    return [example]




def convert_examples_to_features(examples,tokenizer, max_seq_length, max_query_length):
    features = []
    for example in tqdm(examples):
        query_tokens = list(example['question'])
        doc_tokens = example['passage']
        doc_tokens = doc_tokens.replace(u"“", u"\"")
        doc_tokens = doc_tokens.replace(u"”", u"\"")
        start_position = example['gold_span'][0]
        end_position = example['gold_span'][1]

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append("[SEP]")
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        if len(tokens) > max_seq_length:
            tokens[max_seq_length-1] = "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append(
                        {"input_ids":input_ids,
                         "input_mask":input_mask,
                         "segment_ids":segment_ids,
                         "start_position":start_position,
                         "end_position":end_position })
    return features

def train():
    config = JsonConfig('./models/reader/bert_reader/dureader/config.json')
    batch_size,epoch_num,lr,gradient_accumulation_steps,train_paths,dev_paths =\
    config.get_values('batch_size','epoch_num','lr','gradient_accumulation_steps','train_paths','dev_paths')
    pretrained_path,max_q_len,max_seq_len,save_dir,metric_type = config.get_values('pretrained_bert_path','max_q_len','max_seq_len','save_dir','metric_type')
    #check_answer_doc = lambda x:None if DureaderExample(x).illegal_answer_doc() else DureaderExample(x)
    #check_score = lambda x :   x if ('match_scores' not in x.sample_obj) or (x.sample_obj['match_scores'][0]>=0.8) else None
    #span_pre_func = lambda x: x.charspan_preprocessing('gold_span')
    # check whether the  answer span will exceed  the max_seq_len after transform to bert input
    #check_length = lambda x: None if x["gold_span"][1]+3+min(len(x.sample_obj["question"]),max_q_len) >max_seq_len else x
    #flatten_func  = lambda example:example.flatten(['question_id','question','gold_span'],[])
    
    #preprocess_funcs = [ check_answer_doc,check_score,span_pre_func,check_length,flatten_func]
    preprocess_funcs = [lambda x: gold_span_preprocessing_dureader(x,max_q_len,max_seq_len)]
    device = get_default_device()
    model,tokenizer = create_bert_model(pretrained_path,'reader',None,device)
    bert_input_converter = BertInputConverter(tokenizer,max_q_len,max_seq_len)
    decoder = LinearDecoder()
    examples = load_dataset(train_paths, preprocess_funcs)
    #bert_input_converter.convert_examples(examples,'gold_span')
    #optimizer =  SGD(model.parameters(), lr=lr, momentum=0.9)
    examples = convert_examples_to_features(examples,tokenizer, max_seq_len, max_q_len)
    print('Total %d examples'%(len(examples)))
    num_data = len(examples)    
    optimizer = get_bert_optimizer(model,lr=lr,num_data=num_data,batch_size=batch_size,epoch_num=epoch_num,gradient_accumulation_steps=gradient_accumulation_steps )
    #dataset = BertDataset(examples,bert_input_converter,bert_field_names=['gold_span'],device=device)
    dataset = BertDataset(examples,bert_input_converter,bert_field_names=[],device=device)
    if metric_type == 'dureader':
        metric_for_save = 'Bleu-4'
    elif metric_type == 'squad':
        metric_for_save = 'F1'
    highest_metric_for_save = 0
    
    for epcoh in range(epoch_num):
        print('epoch %d'%(epcoh))
        model.train()
        loss_metric = MetricTracer()
        for step,batch in enumerate(dataset.make_batchiter(batch_size)):
            #start_pos,end_pos = tuple(zip(*batch.gold_span))
            #start_pos_t,end_pos_t = torch.tensor(start_pos,device=device, dtype=torch.long),torch.tensor(end_pos,device=device, dtype=torch.long)
            start_pos_t,end_pos_t = torch.tensor(batch.start_position,device=device, dtype=torch.long),torch.tensor(batch.end_position,device=device, dtype=torch.long)
            loss, _, _ = model(batch.input_ids, token_type_ids=batch.segment_ids, attention_mask=batch.input_mask, start_positions=start_pos_t, end_positions=end_pos_t)
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (step+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if (step+1) % 1000 == 0:
                 print('%d step ...'%(step+1))
            loss_metric.add_record(loss.item())
           
            
        loss_metric.print('avg loss is')
        # evaluate 
        model.eval()
        reader = BertReader(model,bert_input_converter,decoder,device)
        evaluator = DureaderMultiDocMrcEvaluator(metric_type)
        print('evaluate')
        result = evaluator.evaluate_reader_on_path(dev_paths,'gold_paragraph',reader)
        print(result)
        if result[metric_for_save] >  highest_metric_for_save:
            print('save best model to %s'%(save_dir))
            model.train()
            highest_metric_for_save = result[metric_for_save]
            torch.save(model.state_dict(), save_dir+"/model.bin")
            model.eval()

if __name__ == "__main__":
    train()
