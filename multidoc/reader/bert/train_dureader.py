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
    answer_doc_idx = source['answer_docs'][0]
    try:
        answer_passage_idx = source['documents'][answer_doc_idx]['most_related_para']
    except:
        return None   
    new_example =  DureaderExample(source).charspan_preprocessing('gold_span').sample_obj
    gold_start,gold_end = new_example['gold_span'][0],new_example['gold_span'][1]
    q_len = min(max_q_len,len(new_example['question'].strip()))
    if q_len+3+gold_end>max_seq_length:
        return None
    return  [{"question_id":new_example['question_id'],"question":new_example['question'].strip(),"passage":new_example["documents"][0]["paragraphs"][0],"gold_span": (gold_start,gold_end)}]



def train():
    config = JsonConfig('./models/reader/bert_reader/dureader/config.json')
    batch_size,epoch_num,lr,gradient_accumulation_steps,train_paths,dev_paths =\
    config.get_values('batch_size','epoch_num','lr','gradient_accumulation_steps','train_paths','dev_paths')
    pretrained_path,max_q_len,max_seq_len,save_dir,metric_type = config.get_values('pretrained_bert_path','max_q_len','max_seq_len','save_dir','metric_type')
    preprocess_funcs = [lambda x: gold_span_preprocessing_dureader(x,max_q_len,max_seq_len)]
    device = get_default_device()
    model,tokenizer = create_bert_model(pretrained_path,'reader',None,device)
    bert_input_converter = BertInputConverter(tokenizer,max_q_len,max_seq_len)
    decoder = LinearDecoder()
    examples = load_dataset(train_paths, preprocess_funcs)
    bert_input_converter.convert_examples(examples,'gold_span')
    print('Total %d examples'%(len(examples)))
    num_data = len(examples)    
    optimizer = get_bert_optimizer(model,lr=lr,num_data=num_data,batch_size=batch_size,epoch_num=epoch_num,gradient_accumulation_steps=gradient_accumulation_steps )
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
            start_pos,end_pos = tuple(zip(*batch.gold_span))
            start_pos_t,end_pos_t = torch.tensor(start_pos,device=device, dtype=torch.long),torch.tensor(end_pos,device=device, dtype=torch.long)
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
