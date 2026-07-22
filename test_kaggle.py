import kagglehub

try:
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("SUCCESS! Dataset downloaded to:", path)
except Exception as e:
    print("FAILED:", e)
