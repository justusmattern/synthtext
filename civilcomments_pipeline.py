from train_model import train_model
from generate import generate
from utils import *
import argparse
import random

def prepare_training_data(args):
    data = dict()
    labels = ['toxic', 'neutral']
    prompted_texts = []

    num_texts = dict()

    for dom in args.domains:
        data[dom] = dict()
        num_texts[dom] = dict()
        for lab in labels:
            with open(f'data/civilcomments/train/{dom}_{lab}.txt', 'r') as file:
                data[dom][lab] = file.readlines()
                num_texts[dom][lab] = len(data[dom][lab])

            for text in data[dom][lab]:
                train_text = ('<BOS> ' + args.prompt.replace('<DOMAIN>', dom).replace('<LABEL>', lab) + ' ' + text + '<EOS>').replace('\n', ' ')
                prompted_texts.append(train_text)

    random.shuffle(prompted_texts)

    return prompted_texts, num_texts


def num_to_generate_by_domain(args, data: dict) -> dict():

    to_generate = dict()

    for dom in args.domains:
        to_generate[dom] = dict()
        label = min(data[dom], key=data[dom].get)
        difference = abs(data[dom['toxic']] - data[dom]['neutral'])
        to_generate[dom]['label'] = label
        to_generate[dom]['num'] = difference
    
    return to_generate



def run(args):

    domain_data_dict, all_prompt_texts = prepare_training_data(args)
    write_texts_to_file(all_prompt_texts, dir=args.prompted_data_dir, filename=f'{args.run_name}_training_data.txt')
    train_model(text_path=f'{args.gen_data_dir}/{args.run_name}_training_data.txt', output_dir=args.model_output_dir, epochs=args.epochs, model_name=args.model, batch_size=args.batch_size)
    to_generate = num_to_generate_by_domain(args, domain_data_dict)

    for dom in args.domains:

        generate(args.model_output_dir,
        prompt='<BOS> ' + args.prompt.replace('<DOMAIN>', dom).replace('<LABEL>', to_generate[dom]['label']),
        num_sequences=to_generate[dom]['num'],
        gen_data_dir=args.gen_data_dir,
        gen_data_filename="{}_{}_{}".format(args.run_name, dom, to_generate[dom]['label'])
        )



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--model', type=str, default='gpt2-xl', help='huggingface model name')
    parser.add_argument('--epochs', type=int, default=3, help='number of finetuning epochs')
    parser.add_argument('--prompt', type=str, default='Write a <LABEL> comment mentioning <DOMAIN> people:')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prompted-data-dir', type = str, default='temp')
    parser.add_argument('--gen-data-dir', type=str, default='generated_data')
    parser.add_argument('--domains', type=str, nargs='+', default=['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white'])
    parser.add_argument('--model-output-dir', type=str, default='trained_model')

    args = parser.parse_args()
    run(args)
