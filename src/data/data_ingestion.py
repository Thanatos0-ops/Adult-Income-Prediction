from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile

def download_dataset():
    try:
        api = KaggleApi()
        api.authenticate()

        dataset_name = 'mosapabdelghany/adult-income-prediction-dataset'
        download_path = 'data/raw/'

        os.makedirs(download_path, exist_ok=True)

        # Download the dataset
        api.dataset_download_files(dataset_name, path=download_path, unzip=False)

        # Find downloaded zip file
        zip_file = os.path.join(download_path, "adult-income-prediction-dataset.zip")

        # Unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)

        print(f"Dataset downloaded and extracted to {download_path }")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_dataset()