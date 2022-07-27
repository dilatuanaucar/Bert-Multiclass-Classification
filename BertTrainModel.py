import csv
import gc
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertModel
import random
import time
import torch.nn.functional as F


def text_preprocessing(text):
    try:
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
        text = re.sub(r'[^\w\text\?]', ' ', text)
        text = re.sub(r'([\;\:\|•«\n])', ' ', text)
        text = re.sub(r'\text+', ' ', text).strip()
        text = text.lower()

    except:
        pass

    if text == None or type(text) != str:
        return ""
    else:
        return text

def preprocessing_for_bert(data, max_len):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text= text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max_len,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BertClassifier(nn.Module):

    def __init__(self, freeze_bert=True):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 3

        self.bert = BertModel.from_pretrained('dbmdz/bert-base-turkish-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        torch.cuda.empty_cache()
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits


def initialize_model(epochs=4):
    bert_classifier = BertClassifier(freeze_bert=False)

    bert_classifier.to(device=torch.cuda.current_device())

    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, epochs=16, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        torch.cuda.empty_cache()
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device=torch.cuda.current_device()) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        avg_train_loss = total_loss / len(train_dataloader)
        print("-" * 70)

        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")
    print("Training complete!")


def evaluate(model, val_dataloader):
    model.eval()
    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        torch.cuda.empty_cache()
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device=torch.cuda.current_device()) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def bert_predict(model, test_dataloader):
    model.eval()
    all_logits = []

    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device=torch.cuda.current_device()) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs

PosNeg= dict()

PosNeg["positiveID"] = [256, 2314, 2613, 2725, 3880, 4024, 4512, 7695, 9232, 9233, 9234]
PosNeg["negativeID"] = [257, 2315, 2614, 2726, 3881, 4025, 4513, 7696, 9235, 9236, 9237]
PosNeg["positiveTextList"] = []
PosNeg["positiveLabelList"] = []
PosNeg["negativeTextList"] = []
PosNeg["negativeLabelList"] = []


def CreateLists():
    AnswerData = pd.read_csv("Train.csv")

    for line in AnswerData.values:
        try:
            questionID=int(line[0])
            answerText = line[1]
            label = int(line[2])

            positiveFlag=False
            negativeFlag=False
            for positiveID in PosNeg["positiveID"]:
                if positiveID==questionID:
                    positiveFlag=True
                    PosNeg["positiveTextList"].append(answerText)
                    PosNeg["positiveLabelList"].append(label)
                    break

            if not positiveFlag:
                for negativeID in PosNeg["negativeID"]:
                    if negativeID == questionID:
                        negativeFlag = True
                        PosNeg["negativeTextList"].append(answerText)
                        PosNeg["negativeLabelList"].append(label)
                        break

            if (not negativeFlag and not positiveFlag):
                print(str(questionID)+" yok!")
        except:
            pass
    return PosNeg["negativeTextList"],PosNeg["negativeLabelList"], PosNeg["positiveTextList"],PosNeg["positiveLabelList"]


if __name__ == '__main__':
    epochs = 50
    batch_size = 4

    negativeTextList,negativeLabelList, positiveTextList,positiveLabelList = CreateLists()

    X = negativeTextList
    y = negativeLabelList

    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', do_lower_case=True)

    all_tweets = negativeTextList
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]
    max_len = max([len(sent) for sent in encoded_tweets])

    token_ids = list(preprocessing_for_bert([X[0]],max_len)[0].squeeze().numpy())
    print('Original: ', X[0])
    print('Token IDs: ', token_ids)

    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_inputs, train_masks = preprocessing_for_bert(X_train,max_len)
    val_inputs, val_masks = preprocessing_for_bert(X_val,max_len)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()

    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(epochs=16)

    gc.collect()
    torch.cuda.empty_cache()

    train(bert_classifier, train_dataloader, val_dataloader, epochs=epochs, evaluation=True)

    resultTime = time.gmtime()
    modelFileName = "MODEL"

    torch.save(bert_classifier,
               str(modelFileName) + "_" +
               str(resultTime.tm_mday) + "." +
               str(resultTime.tm_mon) + "." +
               str(resultTime.tm_year) + "_" +
               str(resultTime.tm_hour) + "." +
               str(resultTime.tm_min) + "." +
               str(batch_size) + ".pth")