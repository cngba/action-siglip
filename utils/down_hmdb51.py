import kagglehub

# Download latest version
path = kagglehub.dataset_download("easonlll/hmdb51")

print("Path to dataset files:", path)
