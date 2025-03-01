from transformers import BertTokenizer, BertModel

# Load a pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
