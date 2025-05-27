# isvl

## Install Environments

Create a new conda environment and install required packages.

```bash
# Create a new conda environment
conda create -n cvprw python=3.8.12

# Activate the environment
conda activate cvprw

# Install required dependencies
pip install -r requirements.txt
```

⚙️ Note: Experiments are conducted on an NVIDIA GeForce RTX 4090 (24GB).
It is recommended to use the same GPU and package versions for best reproducibility. And the CPR result is unstable, please ref to 

##  Prepare Datasets
Unzip the dataset file to the following directory:
```bash
./mvtec_ad_2
```
The structure of the current working directory should be as follows:
```bash
ISVL
├── beit
├── data
├── mvtec_ad_2
│   ├── can
│   ├── fabric
│   ├── ...
```

# Reproduce the result
```bash
bash submitv2.sh
```
The final submission result is the `results.tar.gz` file located in the current directory.

## Acknowledgements

This project is built upon ideas and code from  [**INP-Former**](https://github.com/luow23/INP-Former) and [**CPR**](https://github.com/flyinghu123/CPR).  
We greatly appreciate the authors for making their work open-source and accessible.

If you find this project helpful, please also consider citing their original work.

