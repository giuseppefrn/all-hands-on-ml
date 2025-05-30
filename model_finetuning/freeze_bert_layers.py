from transformers import BertForSequenceClassification, BertTokenizer

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Freeze all parameters first
for param in model.bert.parameters():
    param.requires_grad = False

# Unfreeze last 2 transformer blocks
for i in [10, 11]:  # BERT has 12 layers: 0–11
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = True

# Make sure the classification head is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Optional: Print trainable layers
for name, param in model.named_parameters():
    print(f"{name:55} | Trainable: {param.requires_grad}")