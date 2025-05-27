# isvl
# Create a new conda environment and install required packages.
  conda create -n cvprw python=3.8.12
  conda activate cvprw
  pip install -r requirements.txt

Experiments are conducted on NVIDIA GeForce RTX 4090 (24GB). Same GPU and package version are recommended.

# Prepare Datasets

Unzip the file to ./mvtec_ad_2.

# Reproduce the result
bash submit.sh


# Acknowledgements
We sincerely appreciate INP-Former for its concise, effective, and easy-to-follow approach.
