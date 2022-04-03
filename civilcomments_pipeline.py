from train_model import train_model
import argparse
import random
import os

def prepare_training_data(args):
    data = dict()
    labels = ['toxic', 'neutral']
    prompted_texts = []

    for dom in args.domains:
        data[dom] = dict()
        for lab in labels:
            with open(f'data/civilcomments/train/{dom}_{lab}.txt', 'r') as file:
                data[dom][lab] = file.readlines()

            for text in data[dom][lab]:
                train_text = ('<BOS> ' + args.prompt.replace('<DOMAIN>', dom).replace('<LABEL>', lab) + ' ' + text + '<EOS>').replace('\n', ' ')
                prompted_texts.append(train_text)

    random.shuffle(prompted_texts)

    os.makedirs(args.gen_data_dir, exist_ok=True)
    with open(f'{args.gen_data_dir}/{args.run_name}_training_data.txt', 'w') as f:
        for text in prompted_texts:
            f.write(text+'\n')


def run(args):
    prepare_training_data(args)
    train_model(text_path=f'{args.gen_data_dir}/{args.run_name}_training_data.txt', output_dir=args.model_output_dir, epochs=args.epochs, model_name=args.model, batch_size=args.batch_size)
    generate()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model', type=str, default='gpt2-xl', help='huggingface model name')
    parser.add_argument('--epochs', type=int, default=3, help='number of finetuning epochs')
    parser.add_argument('--prompt', type='str', default='Write a <LABEL> comment mentioning <DOMAIN> people:')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prompted-data-dir', type = str, default='temp')
    parser.add_argument('--gen-data-dir', type=str, default='generated_data')
    parser.add_argument('--domains', type=str, nargs='+', default=['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white'])
    parser.add_argument('--model-output-dir', type=str, default='trained_model')

    args = parser.parse_args()
    run(args)
