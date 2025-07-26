import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/massachusetts-buildings-dataset")

print("Path to dataset files:", path)