
import os 

def write_texts_to_file(texts, dir, filename):
    os.makedirs(dir, exist_ok=True)
    with open(f'{dir}/{filename}_training_data.txt', 'w') as f:
        for text in texts:
            f.write(text+'\n') 
    