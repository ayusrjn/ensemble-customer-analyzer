import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    XLNetTokenizer,
    XLNetForSequenceClassification
)
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
model_paths = {
    'roberta': 'roberta-base1',
    'distilbert': 'path/to/distilbert/model',
    'xlnet': 'path/to/xlnet/model'
}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class EnsembleModel:
    def __init__(self, model_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load RoBERTa
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaForSequenceClassification.from_pretrained(model_paths['roberta'])
        self.roberta_model.to(device)
        
        # Load DistilBERT
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_paths['distilbert'])
        self.distilbert_model.to(device)
        
        # Load XLNet
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.xlnet_model = XLNetForSequenceClassification.from_pretrained(model_paths['xlnet'])
        self.xlnet_model.to(device)
        
        # Set all models to evaluation mode
        self.roberta_model.eval()
        self.distilbert_model.eval()
        self.xlnet_model.eval()
    
    def get_predictions(self, texts, batch_size=16):
        # Create datasets
        roberta_dataset = TextDataset(texts, self.roberta_tokenizer)
        distilbert_dataset = TextDataset(texts, self.distilbert_tokenizer)
        xlnet_dataset = TextDataset(texts, self.xlnet_tokenizer)
        
        # Create dataloaders
        roberta_loader = DataLoader(roberta_dataset, batch_size=batch_size)
        distilbert_loader = DataLoader(distilbert_dataset, batch_size=batch_size)
        xlnet_loader = DataLoader(xlnet_dataset, batch_size=batch_size)
        
        all_predictions = []
        
        with torch.no_grad():
            # Get predictions from each model
            roberta_preds = self._get_model_predictions(self.roberta_model, roberta_loader)
            distilbert_preds = self._get_model_predictions(self.distilbert_model, distilbert_loader)
            xlnet_preds = self._get_model_predictions(self.xlnet_model, xlnet_loader)
            
            # Average predictions
            ensemble_preds = np.mean([roberta_preds, distilbert_preds, xlnet_preds], axis=0)
            
        return ensemble_preds
    
    def _get_model_predictions(self, model, dataloader):
        predictions = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=1)
            predictions.extend(probs.cpu().numpy())
            
        return np.array(predictions)

# Example usage
def predict_from_csv(csv_path, text_column, model_paths, output_path=None):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Initialize ensemble
    ensemble = EnsembleModel(model_paths)
    
    # Get predictions
    predictions = ensemble.get_predictions(df[text_column].values)
    
    # Add predictions to dataframe
    df['prediction'] = predictions.argmax(axis=1)
    df['confidence'] = predictions.max(axis=1)
    
    # Save results if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        
    return df


results = predict_from_csv(
    csv_path='your_input.csv',
    text_column='text',  # replace with your text column name
    model_paths=model_paths,
    output_path='predictions.csv'
)