import torch 
import torch.nn as nn
from encoder import CNNEncoder
from decoder import RNNEncoder

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, train=False):
        super().__init__()
        
        self.encoder = CNNEncoder(embed_size, train)
        self.decoder = RNNEncoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50, device="cuda"):
        result_caption = []
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(0)
            
            states = None
            inputs = torch.tensor([vocabulary.stoi["<SOS>"]]).unsqueeze(0).to(device)
            
            for _ in range(max_length):
                embedded = self.decoder.embed(inputs)
                
                if states is None:
                    embedded = torch.cat((features, embedded), dim=1)
                    
                hidden, states = self.decoder.lstm(embedded, states)
                
                outputs = self.decoder.linear(hidden.squeeze(1))
                predicted = outputs.argmax(1)
                
                result_caption.append(predicted.item())
                inputs = predicted.unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]