import json
import os

def read_block_information(block_dir):
    block_times = {}
    files = list_files(block_dir)

    for f in files:
        with open(f, "r") as file:
            for line in file.readlines():
                obj = json.loads(line)
                
                if 'number' in obj:
                    number = obj['number']
                    timestamp = obj['timestamp']
                    block_times[number] = timestamp
                elif 'block_number' in obj:
                    number = obj['block_number']
                    timestamp = obj['block_timestamp']
                    block_times[number] = timestamp
                else:
                    print("Beware that a block does not have a number or block_number. Line is this:", line)

    return block_times

def list_files(directory_name):
    files = []
    f_list = os.listdir(directory_name)

    for f in f_list:
        file_path = os.path.join(directory_name, f)

        if os.path.isfile(file_path):
            files.append(file_path)
        elif os.path.isdir(file_path):
            files.extend(list_files(file_path))

    return files
