
import gzip
import json

def read_and_write_human_responses(file_path, output_chosen_path, output_human_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as infile:
        with open(output_chosen_path, 'w', encoding='utf-8') as chosen_file:
            with open(output_human_path, 'w', encoding='utf-8') as human_file:
                for line in infile:
                    data = json.loads(line)
                    # Extract the "chosen" key-value
                    chosen_data = {"chosen": data["chosen"]}
                    # Process or print the chosen data as needed
                    # print(chosen_data)
                    # Write the chosen data to the output chosen file
                    chosen_file.write(json.dumps(chosen_data, ensure_ascii=False) + '\n')
                    
                    # Extract and process the human response
                    human_response = data["chosen"].split('\n\n')[1]
                    human_data = {"chosen": human_response}
                    # Process or print the human data as needed
                    # print(human_data)
                    # Write the human data to the output human file
                    human_file.write(json.dumps(human_data, ensure_ascii=False) + '\n')

folder_path = "./hh-rlhf/helpful-base/"
file_name = 'train.jsonl.gz'
file_path = f'{folder_path}/{file_name}'

output_chosen_path = folder_path + 'chosen_' + file_name.replace('.jsonl.gz', '.txt')
output_human_path = folder_path + 'human_' + file_name.replace('.jsonl.gz', '.txt')

read_and_write_human_responses(file_path, output_chosen_path, output_human_path)


