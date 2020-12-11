# MODEL CLASSES!
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

def get_token_classifier_like_output(logits, attention_mask, labels, num_labels):
    outputs = (logits,)
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        outputs = (loss,) + outputs
        
    return outputs
# LSTM-CRF

# simple LSTM

# KIM-CNN-ish with CRF

# simple CNN
#class CNNBasic(nn.Module):
    


# SVM-CRF

# simple SVM with window

class SimpleClassifier(nn.Module):

    def __init__(self, config):        
        super().__init__()
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.max_seq_length, config.max_seq_length*self.num_labels)
        
    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        logits = self.classifier(input_ids.type(torch.FloatTensor))
        
        return get_token_classifier_like_output(logits, attention_mask, labels, self.num_labels)