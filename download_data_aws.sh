mkdir -p ./data

# Download pfam
while true; do
     read -p "Do you wish to download and unzip the pretraining corpus? It is 7.7GB compressed and 19GB uncompressed? [y/n]" yn
     case $yn in
	    [Yy]* ) aws s3 cp s3://proteindata/data_pytorch/pfam.tar.gz .; tar -xzf pfam.tar.gz -C ./data; rm pfam.tar.gz; break;;
            [Nn]* ) exit;;
	    * ) echo "Please answer yes (Y/y) or no (N/n).";;
    esac
done

# Download Vocab/Model files
aws s3 cp s3://proteindata/data_pytorch/pfam.model .
aws s3 cp s3://proteindata/data_pytorch/pfam.vocab .

mv pfam.model data
mv pfam.vocab data

# Download Data Files
aws s3 cp s3://proteindata/data_pytorch/secondary_structure.tar.gz .
aws s3 cp s3://proteindata/data_pytorch/proteinnet.tar.gz .
aws s3 cp s3://proteindata/data_pytorch/remote_homology.tar.gz .
aws s3 cp s3://proteindata/data_pytorch/fluorescence.tar.gz .
aws s3 cp s3://proteindata/data_pytorch/stability.tar.gz .

tar -xzf secondary_structure.tar.gz -C ./data
tar -xzf proteinnet.tar.gz -C ./data
tar -xzf remote_homology.tar.gz -C ./data
tar -xzf fluorescence.tar.gz -C ./data
tar -xzf stability.tar.gz -C ./data

rm secondary_structure.tar.gz
rm proteinnet.tar.gz
rm remote_homology.tar.gz
rm fluorescence.tar.gz
rm stability.tar.gz
