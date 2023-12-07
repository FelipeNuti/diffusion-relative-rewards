import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder_path", type=str, help="Path to folder containing the latent files")
args = parser.parse_args()

folder_path = args.folder_path

# get a list of the file names in the folder
file_names = os.listdir(folder_path)

# filter the list to only include files with the expected format
file_names = [f for f in file_names if f.startswith("latents_") and f.endswith(".pt")]

# sort the list of file names
file_names.sort()

# load the tensors from each file into a list
tensors = []
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    tensor = torch.load(file_path, map_location=torch.device("cpu"))
    print(f"Loaded {file_path}")
    tensors.append(tensor)

print("Finished loading, starting to merge")
# concatenate the tensors along the first dimension
merged_tensor = torch.cat(tensors, dim=0)
print("Merged tensors")

# save the merged tensor to disk
output_file_path = os.path.join(folder_path, "merged.pt")
torch.save(merged_tensor, output_file_path)
