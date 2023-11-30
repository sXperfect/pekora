# High-Performance 3D Genome Reconstruction Using K-th Order Spearman’s Rank Correlation Approximation (PEKORA)

This is the open source software **PEKORA**.

## Quick start

For a smooth quick start, we have created a short guide to run our software
We have tested this software on `Ubuntu` operating system with `conda` software.

First, clone the repository and enter the directory:

```shell
git clone https://github.com/sXperfect/pekora
cd pekora
```

Create a virtual environment using `conda` and install the necessary libraries
```shell
conda create -y -n pekora python=3.11
conda activate pekora
conda install -y -c conda-forge cmake gxx_linux-64 gcc_linux-64 zlib curl
```

Install Python libraries
```shell
pip install -r requirements.txt
pip install hic2cool cooltools cooler hic-straw
```

Download the input file `GSE63525_GM12878_insitu_primary_30.hic` from [GEO GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525):
```shell
mkdir -p data && cd data
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%5F30%2Ehic
```

Our tool supports both `hic` and `mcool` formats, but we strongly recommend using `mcool`.
To convert `hic` data to `mcool`:
```shell
hic2cool convert GSE63525_GM12878_insitu_primary_30.hic GSE63525_GM12878_insitu_primary_30.mcool
```

Go back to the root directory
```
cd ..
```

Run our **PEKORA** tool:
```shell
python main.py +node=default +exp=profile4_cpu args.input=GSE63525_GM12878_insitu_primary_30.mcool args.res=5000 args.chr=\'22\'
```
to run `profile4` on `cpu`, reconstructing from `GSE63525_GM12878_insitu_primary_30.mcool` chromosome `'22'` at resolution `5000`.

All configs are stored in `configs/exp`, consisting of profiles 1 to profile 4:
- `profile1_cpu`
- `profile2_cpu`
- `profile3_cpu`
- `profile4_cpu`

To run on gpu, add the parameter `args.accelerator=gpu` and set the precision to either 16 or 32 bits with the parameter `args.precision=<precision>` (default is 64 bits or double for CPU).

The results are stored in the `results` folder.

## Usage policy

The open source PEKORA is made available before scientific publication.

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

In addition, the `hic2cool`, `cooltools`, `cooler`, and `hic-straw` libraries are required to read the input data in `.hic` and `.mcool` formats and must be installed after the installation of the libraries listed in the [requirements.txt](requirements.txt) file.

## Usage

Input file must be stored in the [data](data) folder.

For the options or configuration, our tool relies on the [Hydra](https://hydra.cc/docs/intro/) library.
Please refer to the [base_config.yaml](configs/base_config.yaml) or the profiles in the [exp](configs/exp) folder to see the full options of our tool.

Reconstructing 3D chromosome structure
```shell
python main.py +node=default +exp=<profile_name> args.input=<input_file> args.res=<resolution> args.chr=\'<chromosome>\' args.balancing=<balancing> [args.accelerator=gpu] args.precision=<precision>

arguments:
  profile_name      Profile filename in the configs/exp folder
  input_file        Input file in the data folder
  resolution        Resolution
  chromosome        Chromosome name
  balancing         Matrix balancing method. Must be precomputed and stored in the input file.
  precision         Floating point precision for the calculation. The valid values are 16, 16-true, bf16-mixed, bf16-true, 32 and 64 (see pytorch lightning precision)
  
options:
  args.accelerator=gpu  Run on GPU (if available and supported)
```

Computing the Spearman's rank correlation of PEKORA result
```shell
usage: python -m pekora comp_pekora_spearmanr [-h] [--chr2_region chr2_region] chr1_region resolution balancing data_file_path points_file_path

positional arguments:
  chr1_region           Region of <chr1>
  resolution            Resolution
  balancing             Name of balancing method
  data_file_path        Input file path
  points_file_path      Points file path

options:
  -h, --help            show this help message and exit
  --chr2_region chr2_region
                        Region of <chr2>
```
Make sure that the `data_file_path` (or input file) is the one used for the reconstruction process.