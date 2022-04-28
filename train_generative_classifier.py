from tqdm import tqdm
import argparse
import torch
from torch.optim import Adam
from torch import nn
from generative_classifier_supervised import CausalClassifier
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds import get_dataset
from sklearn.metrics import accuracy_score, f1_score

def get_data_loaders(training_batch_size, train_ratio, val_ratio, test_ratio):
    dataset = get_dataset(dataset="civilcomments", download=True)

    train_data = dataset.get_subset("train", frac=train_ratio)
    train_loader = get_train_loader("standard", train_data, batch_size=training_batch_size)

    val_data = dataset.get_subset("val", frac=val_ratio)
    val_loader = get_eval_loader("standard", val_data, batch_size=4)

    test_data = dataset.get_subset("test", frac=test_ratio)
    test_loader = get_eval_loader("standard", test_data, batch_size=4)

    return train_data, train_loader, val_data, val_loader, test_data, test_loader


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
    train_data, train_loader, val_data, val_loader, test_data, test_loader = get_data_loaders(args.batch_size, train_ratio=args.train_set_ratio, val_ratio=args.val_set_ratio, test_ratio=args.test_set_ratio)
    model = CausalClassifier(gpt2_model = args.model, gpt2_tokenizer=args.tokenizer, device_id=args.device_id).to(f'cuda:{args.device_id}')
    optimizer = Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.epochs):

        if not args.eval_only:
            model.train()
            all_predictions = []
            all_labels = []
            all_domains = []
            for x, y, meta in tqdm(train_loader):
                domains_list = meta_to_domain(meta)
                loss, label_probs, predictions = model(x, domains_list, y)

                all_predictions.extend(predictions.tolist())
                all_labels.extend(y.tolist())
                all_domains.extend([m for m in meta])

                loss.backward()
                optimizer.step()
            
            print('train results:')
            print(accuracy_score(all_predictions, all_labels))
            print(f1_score(all_predictions, all_labels))

            train_data.eval(torch.LongTensor(all_predictions).cpu(), torch.LongTensor(all_labels).cpu(), torch.stack(all_domains).cpu())
            #torch.save(model.state_dict(), f'gpt2_epoch{epoch}.pt')

        model.eval()
        all_predictions = []
        all_labels = []
        all_domains = []
        for x, y, meta in tqdm(val_loader):
            domains_list = meta_to_domain(meta)
            loss, label_probs, predictions = model(x, domains_list, y)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(y.tolist())
            all_domains.extend([m for m in meta])
        
        print('val results:')
        print(accuracy_score(all_predictions, all_labels))
        print(f1_score(all_predictions, all_labels))

        val_data.eval(torch.LongTensor(all_predictions), torch.LongTensor(all_labels), torch.stack(all_domains))

        model.eval()
        all_predictions = []
        all_labels = []
        all_domains = []
        for x, y, meta in tqdm(test_loader):
            domains_list = meta_to_domain(meta)
            loss, label_probs, predictions = model(x, domains_list, y)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(y.tolist())
            all_domains.extend([m for m in meta])
        
        print('test results:')
        print(accuracy_score(all_predictions, all_labels))
        print(f1_score(all_predictions, all_labels))
        test_data.eval(torch.LongTensor(all_predictions), torch.LongTensor(all_labels), torch.stack(all_domains))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--train-set-ratio', type=float, default=1.0)
    parser.add_argument('--val-set-ratio', type=float, default=1.0)
    parser.add_argument('--test-set-ratio', type=float, default=1.0)
    args = parser.parse_args()
    run(args)


