import os
from . import args
from . datautil import Dureader
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim

from  multidoc.util.bert.optimizer import get_bert_optimizer
from  multidoc.util.bert import create_bert_model,BertDataset,BertInputConverter,BertReader
from multidoc.util.bert.optimizer import BertAdam
from multidoc.util.bert.modeling import BertForQuestionAnswering, BertConfig
from multidoc.util.bert.tokenization import BertTokenizer
from multidoc.eval import  DureaderMultiDocMrcEvaluator
from multidoc.core.op import LinearDecoder



# 随机种子
random.seed(args.seed)
torch.manual_seed(args.seed)






def evaluate(model, dev_data):
    total, losses = 0.0, []
    device = args.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:

            input_ids, input_mask,segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device), \
                                     segment_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device))
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("%s/log"%(args.save_dir), 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)


def train():
    device = args.device
    model,tokenizer = create_bert_model(args.bert_pretrained_dir,'reader',weight_path=None,device=device)
    optimizer = get_bert_optimizer(model,lr=args.learning_rate,num_data=187818,batch_size=args.batch_size,epoch_num=args.num_train_epochs,gradient_accumulation_steps=args.gradient_accumulation_steps )
    # 准备数据
    data = Dureader(args.data_dir)
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 100000.0
    highest_bleu4 = 0.0
    model.train()
    for i in range(args.num_train_epochs):
        for step , batch in enumerate(tqdm(train_dataloader, desc="Epoch")):
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                                        batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            input_ids, input_mask, segment_ids, start_positions, end_positions = \
                                        input_ids.to(device), input_mask.to(device), segment_ids.to(device), start_positions.to(device), end_positions.to(device)

            # 计算loss
            loss, _, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 更新梯度
            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        tokenizer = BertTokenizer('%s/vocab.txt'%(args.bert_pretrained_dir), do_lower_case=True)       
        bert_input_converter = BertInputConverter(tokenizer,args.max_query_length,args.max_seq_length)
        decoder = LinearDecoder()
        model.eval()
        reader = BertReader(model,bert_input_converter,decoder,device)
        evaluator = DureaderMultiDocMrcEvaluator('dureader')
        print('evaluate')
        result = evaluator.evaluate_reader_on_path(['./data/search.dev.json','./data/zhidao.dev.json'],'gold_paragraph',reader)
        print(result)
        if result['Bleu-4'] >   highest_bleu4 :
            print('save best model to %s'%(args.save_dir))
            model.train()
            highest_bleu4  = result['Bleu-4']
            torch.save(model.state_dict(), args.save_dir+"/model.bin")
 
       
                
                
        #eval_loss = evaluate(model, dev_dataloader)
        #if eval_loss < best_loss:
        #    best_loss = eval_loss
        #    torch.save(model.state_dict(), args.save_dir + "/model.bin")
        #    model.train()

if __name__ == "__main__":
    train()
