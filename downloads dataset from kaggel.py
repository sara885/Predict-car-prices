import os
import shutil
import kagglehub



path = kagglehub.dataset_download("taeefnajib/used-car-price-prediction-dataset")
print("Path to dataset files:", path)


target_path = "/content/sample_data/dtcar"


os.makedirs(target_path, exist_ok=True)


for file_name in os.listdir(path):
    full_file_path = os.path.join(path, file_name)
    shutil.copy(full_file_path, target_path)

print("✔️ تم نسخ الملفات إلى:", target_path)
