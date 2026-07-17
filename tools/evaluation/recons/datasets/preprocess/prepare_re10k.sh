# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

# download metadata
mkdir datas/re10k
cd datas/re10k
wget https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz
tar -xvzf RealEstate10K.tar.gz
mv RealEstate10K metadata
rm RealEstate10K.tar.gz
cd ../..

# download re10k test sequences by yourself, you can refer to the below script
python datasets/preprocess/download_re10k.py

# convert camera annotations in metadata to video data folder
python datasets/preprocess/prepare_re10k.py
