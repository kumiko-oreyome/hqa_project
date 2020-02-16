import os
import torch
from torch.optim import SGD
import argparse
from multidoc.util import JsonConfig
from multidoc.util.bert import 
from multidoc.dataset.dureader import load_pointwise_examples


#from  multidoc.eval import DureaderRankingEvaluator





def train():
    config = JsonConfig('./models/ranker/bert_ranker/debug/config.json')
    sample_method,neg_num,batch_size,epoch_num,lr,train_paths,dev_paths =\
    config.get_values('sample_method','neg_num','batch_size','epoch_num','lr','train_paths','dev_paths')
    pretrained_path,max_q_len,max_seq_len,save_dir = config.get_values('pretrained_bert_path','max_q_len','max_seq_len','save_dir')
    device = get_default_device()
    model,tokenizer = create_bert_model(pretrained_path,'ranker',None,device)
    bert_input_converter = BertInputConverter(tokenizer,max_q_len,max_seq_len)
    examples = load_pointwise_examples(train_paths,sample_method,neg_num)
    print('Total %d examples'%(len(examples)))
    optimizer =  SGD(model.parameters(), lr=lr, momentum=0.9)
    dataset = BertDataset(examples,bert_input_converter,device=device)
    highest_precision = 0 
    for epcoh in range(epoch_num):
        print('epoch %d'%(epcoh))
        model.train()
        loss_metric = MetricTracer()
        for step,batch in enumerate(dataset.make_batchiter(batch_size)):
            if (step+1) % 1000 == 0:
                 print('%d step ...'%(step+1))
            label_t = torch.tensor(batch.label,device=device, dtype=torch.long) 
            loss= model(batch.input_ids,batch.segment_ids,batch.input_mask,label_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_metric.add_record(loss.item())

        loss_metric.print('avg loss is')
        # evaluate 
        model.eval()
        reader = BertPointwiseRanker(model,bert_input_converter,device)
        evaluator = DureaderRankingEvaluator('gold_paragraph')
        print('evaluate')
        result = evaluator.evaluate_on_path(dev_paths,sample_method=sample_method,neg_num=neg_num)
        print(result)
        if result['precision'] >  highest_precision:
            print('save best model to %s'%(save_dir))
            model.train()
            highest_precision = result['precision']
            torch.save(model.state_dict(), save_dir+"/model.bin")
            model.eval()
    
if __name__ == '__main__':
    train()