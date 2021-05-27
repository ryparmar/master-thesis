import torch
from torch import nn
import transformers


class Encoder(nn.Module):
    """ a wrapper class for Huggingface transformer models"""
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.encoder = transformers.BertModel.from_pretrained(config.bert_model)
        self.linear = torch.nn.Linear(768, 512)  # from Two-tower paper
        self.config = config

    def forward(self, x, x_mask):
        output = self.encoder(input_ids=x, attention_mask=x_mask)
        # print(output)
        # out.last_hidden_state.shape = torch.Size([4, 85, 768]), where (batch_size, max_seq_length, token embedding size)
        
        last_layer_cls_tokens = output[0][:, 0] #.last_hidden_state == [0]
        cls_out = self.linear(last_layer_cls_tokens)  # selects the cls token's embedding
        # cls_out = self.linear(hidden[:, 0])  # selects the cls token's embedding
        return cls_out
