import os
import time

def print_dict(dictionary):
    assert type(dictionary) == dict
    for key, value in dictionary.items():
        print(f"{key}: {value}")

# def check_required_files(dataset_dir, required_files):
#     existing_files = set(os.listdir(dataset_dir))
#     missing_files = [f for f in required_files if f not in existing_files]

#     assert not missing_files, f"Missing required files: {', '.join(missing_files)}"
#     return True
def print_running_time(start_time):
    running_time = time.time() - start_time
    print(f"Running time: {running_time}s")
    return running_time