
import gzip
import json
import os

def read_and_write_human_responses(file_path, output_chosen_path, output_all_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as infile:
        with open(output_chosen_path, 'w', encoding='utf-8') as chosen_file:
            with open(output_all_path, 'w', encoding='utf-8') as all_file:
                for line in infile:
                    data = json.loads(line)
                    # Extract the "chosen" key-value
                    chosen_data = data["chosen"]
                    # Process or print the chosen data as needed
                    chosen_data = data["chosen"]
                    chosen_file.write(json.dumps({'text': chosen_data}, ensure_ascii=False) + '\n')

                    # Write the 'chosen' data to the all file
                    all_file.write(json.dumps({'text': chosen_data}, ensure_ascii=False) + '\n')

                    # Write the 'rejected' data to the all file on separate rows
                    rejected_entry = data["rejected"]
                    all_file.write(json.dumps({'text': rejected_entry}, ensure_ascii=False) + '\n')


folder_names = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']

# Base path where folders are located
base_path = "./hh-rlhf2/"
splits = ['train', 'test']
for folder in folder_names:
    for split in splits:
        full_folder_path = os.path.join(base_path, folder)
        file_name = f'{split}.jsonl.gz'
        file_path = os.path.join(full_folder_path, file_name)
        print("file_path", file_path)
        output_chosen_path = os.path.join(full_folder_path, 'chosen_' + file_name.replace('.jsonl.gz', '.txt'))
        output_human_path = os.path.join(full_folder_path, 'all_' + file_name.replace('.jsonl.gz', '.txt'))

        # Call the function with the paths for the current folder
        read_and_write_human_responses(file_path, output_chosen_path, output_human_path)


