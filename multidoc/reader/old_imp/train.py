import os
from . import args
from . datautil import Dureader
import torch
import random
import pickle
from tqdm import tqdm
from torch import nn, optim

from multidoc.util.bert.optimizer import BertAdam
from multidoc.util.bert.modeling import BertForQuestionAnswering, BertConfig

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
    # 加载预训练bert
    model = BertForQuestionAnswering.from_pretrained(args.bert_pretrained_dir)
    device = args.device
    model.to(device)

    # 准备 optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=args.num_train_optimization_steps)

    # 准备数据
    data = Dureader(args.data_dir)
    train_dataloader, dev_dataloader = data.train_iter, data.dev_iter

    best_loss = 100000.0
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

            # 验证
            if step % args.log_step == 4:
                eval_loss = evaluate(model, dev_dataloader)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), args.save_dir + "/model.bin")
                    model.train()

if __name__ == "__main__":
    train()
