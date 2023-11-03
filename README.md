# High-Performance 3D Genome Reconstruction Using K-th Order Spearman’s Rank Correlation Approximation (PEKORA)

This is the open-source software **PEKORA**.

## Quick start

For a smooth quick start, we created a short guide to run our software
We have tested this software on `Ubuntu` operating system with `conda` software.

First, clone the repository and enter the directory:

```shell
git clone https://github.com/sXperfect/pekora
cd pekora
```

Create a virtual environment using `conda` and install necessary libraries
```shell
conda create -y -n pekora python=3.11
conda activate pekora
conda install -y -c conda-forge cmake gxx_linux-64 gcc_linux-64 zlib curl
```

Install python libraries
```shell
pip install -r requirements.txt
pip install hic2cool cooltools cooler hic-straw
```

Download input file `GSE63525_GM12878_insitu_primary_30.hic` from [GEO GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525):
```shell
mkdir -p data && cd data
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%5F30%2Ehic
```

Our tool support both `hic` and `mcool` format but we highly recommend using `mcool`.
To convert `hic` data to `mcool`:
```shell
hic2cool convert GSE63525_GM12878_insitu_primary_30.hic GSE63525_GM12878_insitu_primary_30.mcool
```

Go back to the root directory
```
cd ..
```

Run our tool **PEKORA**:
```shell
python main.py +node=default +exp=profile4_cpu args.input=GSE63525_GM12878_insitu_primary_30.mcool args.res=5000 args.chr=\'22\'
```
for running `profile4` on `cpu`, reconstrucing from `GSE63525_GM12878_insitu_primary_30.mcool` chromosome `"22"` at resolution `5000`.

All configs are stored in `configs/exp`, consisting of profile 1 to profile 4:
- `profile1_cpu`
- `profile2_cpu`
- `profile3_cpu`
- `profile4_cpu`

To run on gpu, add parameter `args.accelerator=gpu`

The results are stored inside `results` folder.

## Usage policy

The open-source PEKORA is made available before scientific publication.

This pre-publication software is preliminary and may contain errors.
The software is provided in good faith, but without any express or implied warranties.
We refer the reader to our [license](LICENSE).

The goal of our policy is that early release should enable the progress of science.
We kindly ask to refrain from publishing analyses that were conducted using this software while its development is in progress.

## Dependencies

Python 3.8 or higher is required.
It is recommended that you create a virtual environment using conda.
For conda users, the `cmake`, `gcc`, `zlib`, `curl`, and `gxx` libraries are required and can be installed through:

```shell
conda install -y -c conda-forge cmake gxx_linux-64 gcc_linux-64 zlib curl
```

See [requirements.txt](requirements.txt) for the list of required Python libraries.
