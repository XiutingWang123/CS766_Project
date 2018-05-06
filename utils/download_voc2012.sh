
cd ../data

echo "Downloading Pascal VOC 2012 data..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo "Extracting VOC data..."
tar xf VOCtrainval_11-May-2012.tar

mv VOCdevkit pascal_voc/.

echo "Done."