import torch.nn as nn
from config import CONFIG
from transformers import AutoModel

class BERTRegressor(nn.Module):
    """
    BERT Regressor model for rating prediction
    """
    def __init__(self):
        super(BERTRegressor, self).__init__()

        self.bert = AutoModel.from_pretrained(CONFIG['BERT_MODEL'])
       
        self.regressor = nn.Sequential(
             nn.Dropout(0.3),
             nn.Linear(self.bert.config.hidden_size, 128),
             nn.GELU(),
             nn.Linear(128, 1), 
         )

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        
        return self.regressor(cls).squeeze(-1)