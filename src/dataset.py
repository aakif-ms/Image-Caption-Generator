import torch
from torch.utils.data import Dataset
from PIL import Image


class FlickrDataset(Dataset):
    def __init__(self, caption_file, img_dir, vocab, transform=None, max_length=50):

        self.img_dir = img_dir
        self.caption_file = caption_file
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

        self.data = [
            {
                "image": "1000268201_693b08cb0e.jpg",
                "caption": "A child in a pink dress is climbing up a set of stairs in an entry way .",
            },
            {
                "image": "1001773457_577c3a7d70.jpg",
                "caption": "A black dog and a tri-colored dog playing with each other on the road .",
            },
            {
                "image": "1002674143_1b742ab4b8.jpg",
                "caption": "There is a girl with pigtails sitting in front of a rainbow painting .",
            },
        ]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption = self.data[idx]["caption"]
        img_name = self.data[idx]["image"]

        img = Image.new('RGB', (224, 224), color='blue')
        
        if self.transform:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        while len(numericalized_caption) < self.max_length:
            numericalized_caption.append(self.vocab.stoi["<PAD>"])

        return torch.tensor(img), torch.tensor(numericalized_caption)