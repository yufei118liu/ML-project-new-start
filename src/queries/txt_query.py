import os 



def txt_to_query(file_path):
    with open(file_path, "r") as f:
        queries = []
        for line in f:
            queries.append(line[:-1])
    # print(f"out length")
    return queries