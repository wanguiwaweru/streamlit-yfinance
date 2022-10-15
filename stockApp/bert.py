from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
from typing import Optional
import numpy as np
import torch

class SentimentResponse(BaseModel):
    status_code: Optional[int]
    message: Optional[str]
    sentiment: Optional[list[str]]


labels = ['Cyberbullying', 'Insult', 'Profanity', 'Sarcasm', 'Threat', 'Exclusion', 'Pornography', 'Spam']
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
 
model = AutoModelForSequenceClassification.from_pretrained('./model_save')
tokenizer = AutoTokenizer.from_pretrained('./model_save')


def predictions(text):

    res = SentimentResponse()

    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k,v in encoding.items()}
    outputs = model(**encoding)
    logits = outputs.logits
    #print(logits)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    print(probs)
    predictions[np.where(probs >= 0.5)] = 1
    
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    print(predicted_labels)
    if len(predicted_labels) >=1:
        res.status_code = 200
        res.message = f'This post has sentiments that are {predicted_labels}'
        res.sentiment = predicted_labels
    else:
        res.status_code = 200
        res.message = f'This post has sentiments that are non-threatening.'

    return res

text =  "You are a bitch.go kill yourself"
print(predictions(text))
print(predictions('go kill yourself'))