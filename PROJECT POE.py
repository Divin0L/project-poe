
# Importing the necessary libraries
import pandas as pd
import os
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

# download the stopwords
try:
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

# Load dataset 2
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Data_poe.csv')
try:
    Data2 = pd.read_csv(os.path.join(script_dir, 'Data_poe2.csv'))
except (FileNotFoundError, pd.errors.EmptyDataError) as e:
    print(f"Error loading Data_poe2.csv: {e}")
    Data2 = pd.DataFrame()  # or handle it as needed

# Load dataset 1
try:
    Data = pd.read_csv(data_path)
except (FileNotFoundError, pd.errors.EmptyDataError) as e:
    print(f"Error loading Data_poe.csv: {e}")
    Data = pd.DataFrame()  # or handle it as needed

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Combine the data
all_text_data = Data.astype(str).agg(' '.join, axis=1).tolist()
all_text_data2 = Data2.astype(str).agg(' '.join, axis=1).tolist()

all_text_data = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in all_text_data]
all_text_data2 = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in all_text_data2]

# Tokenizing the data in batches
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def batch_tokenize(text_data, batch_size=100):
    def tokenize_batch(batch):
        return tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length= 300)
    
    tokenized_batches = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(tokenize_batch, text_data[i:i + batch_size]) for i in range(0, len(text_data), batch_size)]
        tokenized_batches = [future.result() for future in futures]
    return tokenized_batches

def combine_batches(batches):
    # Determine the maximum sequence length across all batches
    max_length = max(batch['input_ids'].size(1) for batch in batches)

    # Allocate a tensor with the appropriate size for the longest sequence
    input_ids = torch.zeros((sum(batch['input_ids'].size(0) for batch in batches), max_length), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)  # Assuming you want the same max length for attention_mask

    current_index = 0
    for batch in batches:
        batch_size = batch['input_ids'].size(0)
        seq_len = batch['input_ids'].size(1)

        # Pad the batch to the maximum sequence length
        input_ids[current_index:current_index + batch_size, :seq_len] = batch['input_ids']
        attention_mask[current_index:current_index + batch_size, :seq_len] = batch['attention_mask']

        current_index += batch_size

    return {'input_ids': input_ids, 'attention_mask': attention_mask}

    for batch in batches:
        batch_size = batch['input_ids'].size(0)
        input_ids[current_index:current_index + batch_size] = batch['input_ids']
        attention_mask[current_index:current_index + batch_size] = batch['attention_mask']
        current_index += batch_size

    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Split the data into training and testing datasets
clean_data_batches = batch_tokenize(all_text_data)
clean_data2_batches = batch_tokenize(all_text_data2)

clean_data = combine_batches(clean_data_batches)
clean_data2 = combine_batches(clean_data2_batches)

# Ensure data is valid
if clean_data is None or len(clean_data['input_ids']) == 0 or len(clean_data['attention_mask']) == 0:
    raise ValueError("Clean data is empty or improperly formatted.")

# Create the datasets
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


training_data = TextDataset(clean_data)
testing_data = TextDataset(clean_data2)

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=3, 
    output_dir='./results', 
    per_device_train_batch_size=3, 
    per_device_eval_batch_size=3,
    warmup_steps=500, 
    weight_decay=0.01, 
    logging_dir='./logs', 
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    logging_steps=10,
    eval_steps=500,
    save_steps=500,
    metric_for_best_model='loss',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=testing_data,
    tokenizer=tokenizer
)

trainer.train()
eval_results = trainer.evaluate()
model.train()

outputs = model(
    input_ids=clean_data['input_ids'][:2].to(device),
    attention_mask=clean_data['attention_mask'][:2].to(device),
# Ensure padding tokens in labels are ignored
    labels = clean_data['input_ids'][:2].clone() 
    )
labels[labels == tokenizer.pad_token_id] = -100 
 
# Check if the model outputs a loss
if hasattr(outputs, 'loss'):
    loss = outputs.loss
    print(f"Loss: {loss.item()}")
else:
    print("No loss returned from the model.")