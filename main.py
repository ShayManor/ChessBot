import kagglehub

# Download latest version
path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")

print("Path to dataset files:", path)