# MODEL CLASSES!
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


def get_token_classifier_like_output(logits, attention_mask, labels, num_labels):
    ''' Provides the same output format as BertForTokenClassification.'''
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

def get_outputs_with_kd_loss(outputs, attention_mask, teacher_predictions, kd_param, loss_fct_kd):
    standard_loss = outputs[0]
    student_predictions = outputs[1]
    
    # TODO: take regard to attention mask!!
    # input should be log-probabilities
    student_predictions = F.log_softmax(student_predictions, -1)
    # target should be probabilities
    teacher_predictions = F.softmax(teacher_predictions, -1)
    kd_loss = loss_fct_kd(student_predictions, teacher_predictions)
    total_loss = standard_loss+kd_param*kd_loss
    
    outputs = (total_loss, student_predictions)
    return outputs
    
# LSTM-CRF

# simple LSTM
class SimpleLSTM128(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 128
        self.rnn_size = 128
        self.rnn_depth = 1
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.rnn = nn.LSTM(batch_first=True, input_size=self.embedding_dim, hidden_size=self.rnn_size, 
                          bidirectional=True, num_layers=self.rnn_depth)

        self.top_layer = nn.Linear(2*self.rnn_size, self.num_labels)
              
            
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
        output = self.embedding(input_ids) #(n_seqs, max_len, emb_dim)
                        
        rnn_out, _ = self.rnn(output) #(n_seqs, max_len, 2*rnn_size)
        output = self.top_layer(rnn_out)     
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)
class SimpleLSTM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.rnn_size = 128
        self.rnn_depth = 1
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.rnn = nn.LSTM(batch_first=True, input_size=self.embedding_dim, hidden_size=self.rnn_size, 
                          bidirectional=True, num_layers=self.rnn_depth)

        self.top_layer = nn.Linear(2*self.rnn_size, self.num_labels)
              
            
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
        output = self.embedding(input_ids) #(n_seqs, max_len, emb_dim)
                        
        rnn_out, _ = self.rnn(output) #(n_seqs, max_len, 2*rnn_size)
        output = self.top_layer(rnn_out)     
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)

class WindowSequenceModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()       
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.window_size = 3
        self.device = config.device
        
        # knowledge distillation params
        self.teacher_model = config.teacher_model
        self.loss_fct_kd = config.loss_fct_kd
        self.kd_param = config.kd_param
                 
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.top_layer = nn.Linear(self.window_size*self.embedding_dim, self.num_labels)
                        
    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        teacher_predictions=None
    ):
        output = self.embedding(input_ids)
        n_sent, _, emb_dim = output.shape
        zero_pad = torch.zeros(n_sent, 1, emb_dim, device=self.device)
        word_before_repr = torch.cat([zero_pad, output[:,:-1,:]], dim=1)
        word_after_repr = torch.cat([output[:,1:,:], zero_pad], dim=1)
        
        # combine the 3 embedding tensors
        window_repr = torch.cat([word_before_repr, output, word_after_repr], dim=2)
        
        output = self.top_layer(window_repr)
        
        outputs = get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)
        if self.kd_param == 0 or not self.training:
            return outputs
        else:
            #with torch.no_grad():
                # should fix such that we only need to fetch teacher predictions once
                #teacher_predictions = self.teacher_model(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states)["logits"]
            return get_outputs_with_kd_loss(outputs, attention_mask, teacher_predictions, self.kd_param, self.loss_fct_kd)

# KIM-CNN-ish with CRF
class MultipleWindowCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.cnn_kernel_nbr_pieces = 3
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=self.num_labels, 
                               kernel_size=(3, self.embedding_dim), 
                               padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=1, 
                               out_channels=self.num_labels, 
                               kernel_size=(5, self.embedding_dim), 
                               padding=(2,0))
        self.conv3 = nn.Conv2d(in_channels=1, 
                               out_channels=self.num_labels, 
                               kernel_size=(7, self.embedding_dim), 
                               padding=(3,0))
        
        self.final_conv = nn.Conv1d(in_channels=self.num_labels, 
                               out_channels=self.num_labels, 
                               kernel_size=3, 
                               padding=0,
                               stride=3)
        
        self.final_pool = nn.AvgPool1d(kernel_size=3,
                                       stride=3,
                                       padding=0)

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
        output = self.embedding(input_ids).unsqueeze(1) #(32, 1, 128, 16)
        output = (F.relu(self.conv1(output)).squeeze(3), 
                  F.relu(self.conv2(output)).squeeze(3), 
                  F.relu(self.conv3(output)).squeeze(3)) #(32, 9, 128)*3
        output = torch.cat(output, 2) #(32, 9, 384)
        #output = self.final_conv(output) #(32, 9, 128)
        output = self.final_pool(output) #(32, 9, 128)
        
        output = torch.transpose(F.relu(output), 1, 2).contiguous() #(32, 128, 9)
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)

class MultipleWindowCNN2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.cnn_kernel_nbr_pieces = 3
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=self.num_labels, 
                               kernel_size=(3, self.embedding_dim), 
                               padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=1, 
                               out_channels=self.num_labels, 
                               kernel_size=(5, self.embedding_dim), 
                               padding=(2,0))
        
        self.final_conv = nn.Conv1d(in_channels=self.num_labels, 
                               out_channels=self.num_labels, 
                               kernel_size=3, 
                               padding=0,
                               stride=3)
        
        self.final_pool = nn.AvgPool1d(kernel_size=2,
                                       stride=2,
                                       padding=0)

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
        output = self.embedding(input_ids).unsqueeze(1) #(32, 1, 128, 16)
        output = (F.relu(self.conv1(output)).squeeze(3), 
                  F.relu(self.conv2(output)).squeeze(3)) #(32, 9, 128)*2
        output = torch.cat(output, 2) #(32, 9, 256)
        #output = self.final_conv(output) #(32, 9, 128)
        output = self.final_pool(output) #(32, 9, 128)
        
        output = torch.transpose(F.relu(output), 1, 2).contiguous() #(32, 128, 9)
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)

# simple CNN (softmax)
class SimpleCNNSoftmax(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.cnn_kernel_nbr_pieces = 3
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.cnn = nn.Conv2d(in_channels=1, 
                             out_channels=self.num_labels, 
                             kernel_size=(self.cnn_kernel_nbr_pieces, self.embedding_dim), 
                             padding=(int(self.cnn_kernel_nbr_pieces/2),0))

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
        output = self.embedding(input_ids).unsqueeze(1) #(32, 1, 128, 16)
        output = self.cnn(output).squeeze(3) #(32, 9, 128)
        output = torch.transpose(F.softmax(output, dim=1), 1, 2).contiguous() #(32, 128, 9)
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)

# simple CNN (no softmax)
class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding_dim = 16
        self.cnn_kernel_nbr_pieces = 3
        
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                      embedding_dim=self.embedding_dim, 
                                      padding_idx=config.pad_token_id)
        
        self.cnn = nn.Conv2d(in_channels=1, 
                             out_channels=self.num_labels, 
                             kernel_size=(self.cnn_kernel_nbr_pieces, self.embedding_dim), 
                             padding=(int(self.cnn_kernel_nbr_pieces/2),0))

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
        output = self.embedding(input_ids).unsqueeze(1) #(32, 1, 128, 16)
        output = self.cnn(output).squeeze(3) #(32, 9, 128)
        output = torch.transpose(F.relu(output), 1, 2).contiguous() #(32, 128, 9)
        
        return get_token_classifier_like_output(output, attention_mask, labels, self.num_labels)

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