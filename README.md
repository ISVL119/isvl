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
It is recommended to use the same GPU and package versions for best reproducibility. 


## Additional Notes on Disk Usage and Runtime

The entire project folder will temporarily occupy approximately 110 GB of disk space (excluding the original dataset compressed file of about 30 GB).
And total runtime is around 6 hours.

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

> ⚠️ **Note on CPR Results:**  
> The results from [CPR](https://github.com/flyinghu123/CPR) are not fully stable.  
> We have made every effort to ensure consistency, but there may still be a variation of approximately **0.5%** for each class(fruit_jelly and vial).  
> For more details, please refer to [this issue](https://github.com/flyinghu123/CPR/issues/21) in the CPR repository.

> ⚠️ **Post-processing Errors (Low Probability)**  
> We have run the code multiple times and discovered a low-probability issue during post-processing.  Occasionally, a `"permission denied"` error was observed in the `5_post_image_process_wallnuts.py` script.  This issue occurs **after** the `test_private_mixed` and `test_private` folders have been deleted.  The newly generated folders `test_private_mixed_new` and `test_private_new` **cannot be renamed correctly**, which eventually leads to a failure in the final compression step.  
>  
> If this happens, please check and correct the folder structure and naming under:  
>
> ```
> results/anomaly_images_thresholded/wallnuts/
> ```


## Acknowledgements

This project is built upon ideas and code from  [**INP-Former**](https://github.com/luow23/INP-Former) and [**CPR**](https://github.com/flyinghu123/CPR).  
We greatly appreciate the authors for making their work open-source and accessible.

If you find this project helpful, please also consider citing their original work.

