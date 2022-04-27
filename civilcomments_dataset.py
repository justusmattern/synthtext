import torch
import transformers
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader



class CivilCommentsDataset(torch.utils.data.Dataset):
    def __init__(self, gen_data_path='generated_data', experiment_name= 'experiment2', domains=['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']):
        labels = ['toxic', 'neutral']
        self.texts = []
        self.labels = []

        dataset = get_dataset(dataset="civilcomments", download=True)
        train_data = dataset.get_subset("train")
        train_loader = get_train_loader("standard", train_data, batch_size=1)

        for x, y, meta in train_loader:
            self.texts.append(x[0])
            self.labels.append(y[0].item())

        print('1 count', self.labels.count(1))
        print('0 count', self.labels.count(0))

        for l in labels:
            for d in domains:
                try:
                    with open(f'{gen_data_path}/{experiment_name}_{d}_{l}', 'r') as f:
                        lines = f.read().split('#####')
                        for line in lines:
                            sample = line.split('<BOS>')[0]
                            lab = 1 if l=='toxic' else 0
                            self.texts.append(sample)
                            self.labels.append(lab)
                except FileNotFoundError as e:
                    print(e)

        print('1 count', self.labels.count(1))
        print('0 count', self.labels.count(0))


    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        x = self.texts[index]
        y = self.labels[index]

        return x, y



def get_training_loader_synthdata(batch_size=8, gen_data_path='generated_data', domains=['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']):
    training_set = CivilCommentsDataset(gen_data_path=gen_data_path, domains=domains)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    return training_generator
