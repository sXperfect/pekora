# High-Performance 3D Genome Reconstruction Using K-th Order Spearman’s Rank Correlation Approximation (PEKORA)

**PEKORA** is an open-source, high-performance tool for 3D genome reconstruction.

## Quick start

To get started quickly, we provide a step-by-step guide to run our software. 
We have tested it on `Ubuntu` with `conda` and encourage users to follow these instructions.

First, clone the repository and navigate to the directory:

```shell
git clone https://github.com/sXperfect/pekora
cd pekora
```

Create a virtual environment using `conda` and install required libraries:

```shell
conda create -y -n pekora python=3.11
conda activate pekora
conda install -y -c conda-forge cmake gxx_linux-64 gcc_linux-64 zlib curl
```

Install Python libraries:
```shell
pip install -r requirements.txt
pip install hic2cool cooltools cooler hic-straw
```

Download the input file `GSE63525_GM12878_insitu_primary_30.hic` from GEO [GSE63525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525):

```shell
mkdir -p data && cd data
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%5F30%2Ehic
```

Our tool supports both `hic` and `mcool` file formats.
However, we strongly recommend using `mcool` for optimal performance.
To convert `hic` data to `mcool`:
```shell
hic2cool convert GSE63525_GM12878_insitu_primary_30.hic GSE63525_GM12878_insitu_primary_30.mcool
```

Go back to the root directory
```
cd ..
```

Run **PEKORA**:
```shell
python main.py +node=default +exp=profile1_cpu args.input=GSE63525_GM12878_insitu_primary_30.mcool args.res=5000 args.chr=\'22\'
```
to run `profile1` on `cpu`, reconstructing from `GSE63525_GM12878_insitu_primary_30.mcool` chromosome `'22'` at resolution `5000`.

_Note:_ If the input file is a Hi-C file and an error occurs, "ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found," please update your environment variable by adding the necessary configuration:
```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_conda_installation>/envs/pekora/lib
```

All configs are stored in `configs/exp`, consisting of profiles 1 to profile 2:
- `mds-profile1-cpu`
- `mds-profile2-cpu`

To run on gpu, add the parameter `args.accelerator=gpu` and set the precision to either 16 or 32 bits with the parameter `args.precision=<precision>` (default is 64 bits or double for CPU).

The results are stored in the `results` folder.

## Usage policy

**PEKORA** is an open-source, pre-publication software made available before scientific publication.

This preliminary software may contain errors and is provided in good faith, but without any express or implied warranties.
Please refer to our [license](LICENSE) for more information.

Our goal is to facilitate scientific progress through early release. 
We kindly ask users to refrain from publishing analyses conducted using this software while its development is ongoing.

## Dependencies

Python 3.8 or higher is required.
We recommend creating a virtual environment using `conda`.
For conda users, the `cmake`, `gcc`, `zlib`, `curl`, and `gxx` libraries are required and can be installed through:

```shell
conda install -y -c conda-forge cmake gxx_linux-64 gcc_linux-64 zlib curl
```

Please refer to [requirements.txt](requirements.txt) for the list of required Python libraries.

In addition, the `hic2cool`, `cooltools`, `cooler`, and `hic-straw` libraries are required to read the input data in `.hic` and `.mcool` formats and must be installed after the installation of the libraries listed in the [requirements.txt](requirements.txt) file.

## Usage

Input file must be stored in the [data](data) folder.

**PEKORA** relies on the [Hydra](https://hydra.cc/docs/intro/) library for options and configurations.
Please refer to the [base_config.yaml](configs/base_config.yaml) or the profiles in the [exp](configs/exp) folder for the full list of options.

### Reconstructing 3D chromosome structure
```shell
python main.py +node=default +exp=<profile_name> args.input=<input_file> args.res=<resolution> args.chr=\'<chromosome>\' args.balancing=<balancing> [args.accelerator=gpu] args.precision=<precision>

arguments:
  profile_name      Profile filename in the configs/exp folder
  input_file        Input file in the data folder. Either .hic or .mcool
  resolution        Resolution
  chromosome        Chromosome name
  balancing         Matrix balancing method. Must be precomputed and stored in the input file.
  precision         Floating point precision for the calculation. The valid values are 16, 16-true, bf16-mixed, bf16-true, 32 and 64 (see pytorch lightning precision)
  
options:
  args.accelerator=gpu  Run on GPU (if available and supported)
```

### Computing the Spearman's rank correlation of PEKORA result
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

## Relevant data to reproduce the experiment

The data can be dowloaded from [NCBI](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525).

## Contact

Yeremia Gunawan Adhisantoso <[adhisant@tnt.uni-hannover.de](mailto:adhisant@tnt.uni-hannover.de)>

Fabian Müntefering <[muenteferi@tnt.uni-hannover.de](mailto:muenteferi@tnt.uni-hannover.de)>

Jan Voges <[voges@tnt.uni-hannover.de](mailto:voges@tnt.uni-hannover.de)>