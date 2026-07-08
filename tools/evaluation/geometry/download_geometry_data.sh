export BASE_DATA_DIR=./datas/geometry_data  # Set target data directory

# Depth
wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR}/evaluation_depth_dataset https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/

# Normals
wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/marigold_normals/evaluation_dataset.zip
unzip ${BASE_DATA_DIR}/evaluation_dataset.zip -d ${BASE_DATA_DIR}/evaluation_normal_dataset
rm -f ${BASE_DATA_DIR}/evaluation_dataset.zip