# Reference: https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md
# The scripts provided here are for reference only. Please ensure you have obtained the necessary licenses from the original dataset providers before proceeding.

mkdir -p datas
cd datas
# offered by spann3r, if it is not available, you can see https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md for help
gdown 17qte59UNEW-kwza0FfVO8jD6hbWGVtwG
unzip dtu_test_mvsnet.zip

mv dtu_test_mvsnet_release dtu
rm dtu_test_mvsnet.zip
cd ..
