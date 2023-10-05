# Infectio

Agent-based modeling for Virological Plaque.

![sample of simulation visualization](./attachments/sample_simulation.gif)

# How to use

## Install dependecies
Conda: `conda create env -f environment.yml`.

[Poetry](https://python-poetry.org/) is needed. Run `poetry install` to create python virtual environment. To activate
the environment either use `poetry run python <.py file>` or activate a shell environment using `poetry shell`.

venv: (Could not make conda nor poetry work on hemera, using venv instead) `python -m venv ./venv; source
./venv/bin/activate; pip install -r requirements.txt;`

## Running examples
Either with infectio installed or CWD at root of the project, call `python run.py` for any model in the examples folder.

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