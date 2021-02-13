# SeamCarving

How to run?

Clone this repo via git `git clone https://github.com/MariaKokshaikina/SeamCarving.git`

Create virtual env and install requirements via pip 
`pip install -r requirements.txt`

Create `config.yml` file with the same structure as in `example_config.yml` 
```yaml
dataset_dir: path to dir with images you want to process
output_dir: path to output dir
gifs_dir: path to dir for saving gifs
stats_dir: path to dir for saving stats

pool_processes: number of pool processes for parallel running
```

Check out `Experiments.ipynb` notebook to work in Jupyter with visual examples or `experiments_different_images.py` if you want to process dozen of images/sizes. 