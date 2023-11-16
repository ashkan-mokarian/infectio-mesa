# Infectio

Agent-based modeling for Virological Plaque.

![sample of simulation visualization](./attachments/sample_simulation.gif)

# How to use

## Install dependecies

### Conda
`conda create env -f environment.yml`.

### venv
Could not make conda nor poetry work on hemera, using venv instead: 
```shell
python -m venv ./venv;
source ./venv/bin/activate;
pip install -r requirements.txt;
```

## Run simulation

### VACV + Epithelial model

Create a config file (e.g. similiar to [default config](./infectio/models/vacv/default_config.ini)) and modify
accordingly. You can also use arguments in CLI instead (take a look at `--help` for available options). Then with
python environment active, run: `python(3) infectio/models/vacv/run.py -c path/to/config.py [optional arguments]`.

## Create a new model
TBD

# Personal notes

* Can we avoid discretization of particle scalar fields and work completely with
well defined function. For example solve the diffusion with it, and also for
gradient computations be able to use [this python package: numdifftools](https://numdifftools.readthedocs.io/en/latest/index.html).

* Assumption in code: x,y coords start from lower left. particle variable u also same, i.e. rows correspond with x and
columns with y, first row is x=0 and first column is y=0. Therefore for ploting u, need to do
plt.imshow(u.T, origin='lower')

* Now, molecule released at int(x, y). Could also release with sub-pixel precision but not implemented.

* More accurate gradients? Look here [Structure tensor and diagonal gradients](https://bartwronski.com/2021/02/28/computing-gradients-on-grids-forward-central-and-diagonal-differences/)