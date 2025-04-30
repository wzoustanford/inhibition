import gzip
from pathlib import Path
import re
from nltk.stem import PorterStemmer

from contraction_dict import CONTRACTION_DICT

ps = PorterStemmer()

def decompress_gz_to_txt(gz_path: Path, txt_path: Path) -> None:
    """Decompress a .gz file and write its content to a .txt file."""
    with gzip.open(gz_path, 'rt', encoding='utf8') as f_in, open(txt_path, 'w', encoding='utf8') as f_out:
        for line in f_in:
            f_out.write(line)

def split_text_file(source_path: Path, train_path: Path, valid_path: Path, train_ratio: float = 0.5):
    """Split a text file into training and validation sets."""
    # Read all lines from the source text file
    with open(source_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    split_index = int(total_lines * train_ratio)
    
    train_lines = lines[:split_index]
    valid_lines = lines[split_index:]
    
    with open(train_path, 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)
    
    with open(valid_path, 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)
        
    print(f"Training set: {len(train_lines)} lines, Validation set: {len(valid_lines)} lines.")

# Define the file paths
# gz_file = Path("./data/news.2024.en.shuffled.deduped.gz")
# source_file = Path("./data/news.2024.en.shuffled.deduped.txt")
train_file = Path("./data/news.2024.en.train.txt")
valid_file = Path("./data/news.2024.en.valid.txt")

# Decompress the .gz file and split the text data into training and validation sets
# decompress_gz_to_txt(gz_file, source_file)
# split_text_file(source_file, train_file, valid_file)


# Data preprocessing for LLM, for both training and validation sets

def preprocess_data_txt(input_path: Path, output_path: Path) -> None:
    """Preprocess a text file and write the preprocessed data to another text file."""
    with open(input_path, 'r', encoding='utf8') as f_in , open(output_path, 'w', encoding='utf8') as f_out:
        for line in f_in:
            line = line.strip()
            
            # 1. Convert to lowercase
            line = line.lower()

            # 2. Replace numbers with <NUM> (if a number exist in the line)
            line = re.sub(r'\d+', '<NUM>', line)

            # 3. Stemming
            words = line.split(" ")
            stemmed_words = [ps.stem(word) for word in words]
            line = " ".join(stemmed_words)

            # 4. Replace contraction with full form
            for key in CONTRACTION_DICT:
                line = line.replace(key, CONTRACTION_DICT[key])
            
            # 5. Pattern replacement: replace 's with ""
            line = re.sub(r"'s", '', line)

            # 6. Remove punctuations
            line = re.sub(r'[^\w\s]', '', line)

            # Write the preprocessed line to the output file
            f_out.write(line + "\n")

# Preprocess the training and validation data
preprocess_data_txt(train_file, Path("./data/news.2024.en.train.preprocessed.txt"))
print("DONE TRAINING PREPROCESSING")
preprocess_data_txt(valid_file, Path("./data/news.2024.en.valid.preprocessed.txt"))
print("DONE VALIDATION PREPROCESSING")
