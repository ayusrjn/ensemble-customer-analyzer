from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch 

model1path = "roberta-base1"

model1 = AutoModelForSequenceClassification.from_pretrained(model1path)

tokenizer1 = AutoTokenizer.from_pretrained("roberta-base")

input = "this was not a pleasent movie"

inputs = tokenizer1(input, return_tensors='pt', truncation=True, padding=True)
model1.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1.to(device)

inputs ={key: value.to(device) for key , value in inputs.items()}

with torch.no_grad():
    outputs = model1(**inputs)

logits = outputs.logits

predicted_class = torch.argmax(logits, dim=-1).item()

print(f"Predicted class: {predicted_class}")
print(device)

