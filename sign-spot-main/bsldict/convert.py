import json

# Read data from the text file
with open('/home/uday/SignSpotting/bsldict/bsldict/class', 'r') as txt_file:
    lines = txt_file.readlines()

# Parse the lines and extract words and indices
data = []
for line in lines:
    parts = line.split(', ')
    word = parts[0].split(': ')[1]
    index = int(parts[1].split(': ')[1])
    data.append({"Word": word, "Index": index})

# Writing data to a JSON file
with open('word_index.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)

print("JSON file created successfully.")
