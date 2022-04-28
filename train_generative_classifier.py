import argparse
import torch
from torch.optim import Adam
from torch import nn
from generative_classifier_supervised import CausalClassifier
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds import get_dataset

def get_data_loaders(training_batch_size):
    dataset = get_dataset(dataset="civilcomments", download=True)

    train_data = dataset.get_subset("train")
    train_loader = get_train_loader("standard", train_data, batch_size=training_batch_size)

    val_data = dataset.get_subset("val")
    val_loader = get_eval_loader("standard", val_data, batch_size=1)

    test_data = dataset.get_subset("test")
    test_loader = get_eval_loader("standard", test_data, batch_size=1)

    return train_loader, val_loader, test_loader


def meta_to_domain(meta):
    domain_list = []
    domain_verbalizers = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'differently religious', 'black', 'white']

    for dom_list in meta:
        domains = []
        for i in range(8):
            if dom_list[i] == 1:
                domains.append(domain_verbalizers[i])

        domain_list.append(domains)
    
    return domain_list


def run(args):
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size)
    model = CausalClassifier()
    optimizer = Adam(model.parameters, lr=1e-5)

    for epoch in range(args.epochs):

        if not args.eval_only:
            model.train()
            all_predictions = []
            all_labels = []
            all_domains = []
            for x, y, meta in train_loader:
                domains_list = meta_to_domain(meta)
                loss, label_probs, predictions = model(x, domains_list, y)

                all_predictions.extend(predictions.tolist())
                all_labels.extend(y.tolist())
                all_domains.extend([m for m in meta])

                loss.backward()
                optimizer.step()
            
            train_data.eval(torch.LongTensor(all_predictions), torch.LongTensor(all_labels), torch.stack(all_domains))

        model.eval()
        all_predictions = []
        all_labels = []
        all_domains = []
        for x, y, meta in val_loader:
            domains_list = meta_to_domain(meta)
            loss, label_probs, predictions = model(x, domains_list, y)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(y.tolist())
            all_domains.extend([m for m in meta])
        
        val_data.eval(torch.LongTensor(all_predictions), torch.LongTensor(all_labels), torch.stack(all_domains))

        model.eval()
        all_predictions = []
        all_labels = []
        all_domains = []
        for x, y, meta in test_loader:
            domains_list = meta_to_domain(meta)
            loss, label_probs, predictions = model(x, domains_list, y)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(y.tolist())
            all_domains.extend([m for m in meta])
        
        test_data.eval(torch.LongTensor(all_predictions), torch.LongTensor(all_labels), torch.stack(all_domains))
    
    



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-only', type=bool, action='store_true')

    args = parser.parse_args()
    run(args)


