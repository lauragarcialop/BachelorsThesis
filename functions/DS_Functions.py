import torch
import json
from tensordict import TensorDict

def write_dataset(filename, labels, data):

    samples = {'labels' : labels, 'data' : data}

    # Convert tensors to lists
    for key, value in samples.items():
        if isinstance(value, torch.Tensor):
            samples[key] = value.tolist()
    
    # Write to JSON file
    print(f"Data saved in {filename}")
    with open(filename + ".json", "w") as f:
        json.dump(samples, f)

    return


def read_dataset(filename):
    
    with open(filename, 'r') as f:
        data = json.load(f)

    tensor_dict = TensorDict()
    for key, value in data.items():
        if isinstance(value, list):
            tensor_dict[key] = torch.tensor(value)
        else:
            tensor_dict[key] = value

    return tensor_dict['labels'], tensor_dict['data']