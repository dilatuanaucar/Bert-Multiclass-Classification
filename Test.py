import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer
from BertTrainModel import  TensorDataset
import torch.nn.functional as F

data = pd.read_csv("Test.csv")
negativeQuestion = data[
    (data["questionID"] == 257.0) | (data["questionID"] == 2315.0) | (data["questionID"] == 2614.0) | (
            data["questionID"] == 2726.0) | (data["questionID"] == 3881.0) | (data["questionID"] == 4025.0) | (
            data["questionID"] == 4513.0) | (data["questionID"] == 7696.0) | (data["questionID"] == 9235.0) | (
            data["questionID"] == 9236.0) | (data["questionID"] == 9237.0)]

negativeQuestion = negativeQuestion.reset_index()
negativeQuestion.drop("index", axis=1, inplace=True)
negModel = torch.load("MODEL.pth")
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', do_lower_case=True)


# positiveQuestion = data[
#     (data["questionID"] == 256.0) | (data["questionID"] == 2314.0) | (data["questionID"] == 2613.0) | (
#             data["questionID"] == 2725.0) | (data["questionID"] == 3880.0) | (data["questionID"] == 4024.0) | (
#             data["questionID"] == 4512.0) | (data["questionID"] == 7695.0) | (data["questionID"] == 9232.0) | (
#             data["questionID"] == 9233.0) | (data["questionID"] == 9234.0)]
#
#
# positiveQuestion = positiveQuestion.reset_index()
# positiveQuestion.drop("index", axis=1, inplace=True)
# posModel = torch.load("MODEL.pth")
# tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', do_lower_case=True)


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


def preprocessing_for_bert(data, tokenizer):
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []
    for sent in data:
        if sent != None or sent != "":
            encoded_sent = tokenizer.encode_plus(
                truncation=True,
                text=text_preprocessing(sent),
                add_special_tokens=True,
                max_length=256,
                pad_to_max_length=True,
                return_attention_mask=True
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


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


if __name__ == '__main__':

    inputs, masks = preprocessing_for_bert(negativeQuestion.answerText, tokenizer)
    dataset = TensorDataset(inputs, masks)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)

    probs = bert_predict(negModel, dataloader)
    score = len(probs)

    for probIndex in range(len(probs)):

        element = probs[probIndex]

        realLabel = int(negativeQuestion['label'][probIndex])

        if realLabel != np.argmax(element):
            print("Real Label : ", str(realLabel), "\t Predict Label : ", str(np.argmax(element)), "\t NO MATCH !")
            print("Text : ", negativeQuestion['answerText'][probIndex])
            score -= 1
    print("SCORE : ", score / len(probs))
    print("TOTAL LEN : ", len(probs))
