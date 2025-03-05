import gzip
from pathlib import Path

def decompress_gz_to_txt(gz_path: Path, txt_path: Path) -> None:
    """Decompress a .gz file and write its content to a .txt file."""
    with gzip.open(gz_path, 'rt', encoding='utf8') as f_in, open(txt_path, 'w', encoding='utf8') as f_out:
        for line in f_in:
            f_out.write(line)

def split_text_file(source_path: Path, train_path: Path, valid_path: Path, train_ratio: float = 0.5):
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
gz_file = Path("./data/news.2024.en.shuffled.deduped.gz")
source_file = Path("./data/news.2024.en.shuffled.deduped.txt")
train_file = Path("./data/news.2024.en.train.txt")
valid_file = Path("./data/news.2024.en.valid.txt")

# Decompress the .gz file and split the text data into training and validation sets
decompress_gz_to_txt(gz_file, source_file)
split_text_file(source_file, train_file, valid_file)
