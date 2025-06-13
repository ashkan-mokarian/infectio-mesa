# Infectio

Agent-based modeling for Virological Plaque.

![sample of simulation visualization](./attachments/sample_simulation.gif)

The image above shows the output of a simulation run. (top-left) shows spatial
positions of infected (green) and susceptible (blue) cells, (top-right) shows
concentrations of particles and molecules produced by the system in the same
spatial space as top-left, and (three bottom rows) show metrics produced by the
experiment, which are the radius of the plaque, average radial velocity, and
susceptible and infected population sizes, from top to bottom. These metrics,
possibly with others, are used to evaluate simulation with a reference which is
obtained from real world plaque experiements.

# How to use

## Install dependecies

<!-- ### Conda
`conda create env -f environment.yml`. -->

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
accordingly. You can also use arguments in CLI instead (type `--help` for available options). Then with
python environment active, run: `python(3) infectio/models/vacv/run.py -c path/to/config.py [optional arguments]`. Note
that this config file contains all the parameter settings for the full model.

If you only intend to simulate the `dVGFdF11` case, then it is recommended
to use the config file [dVGFdF11_default_config.ini](./examples/vacv-epithelial/dVGFdF11_default_config.ini)
instead. This config file only contains relevant settings and turns off irrelevant parts of the simulation.

## Create a new model
TBD

# Batch run on SLURM cluster

## Relevant files

* Firsly, using 'scripts/generate_param_list_file_for_array.py' generate a .txt containing a list of experiments where 
each line contains one simulation experiment. By referencing the line number, job descriptions are sent to the array
job scheduler.

* scripts/submit_param_array_from_listfile.sh: Define the specifications of the batch of simulations here such as
number of samples for each parameter, save root, and the parameter combination list generated in the previous step.
This will create a tmp_run.sbatch by populating placeholders in the file scripts/run_array_template.sh, create a new
scripts/tmp_run.sbatch file with the new values filled in and run this file in the end.

* in 'scripts/tmp_run.sbatch' (or originally in scripts/run_array_template.sbatch), there is an offset parameter. In 
the case that the number of experiments exceeds the max jobs per user limit on the cluster, you can change this offset
to go over all the lines in the parameter_combinations.txt. manually change the offset and run tmp_run.sbatch otherwise
no need to run tmp_run.sbatch manually.

# Notes

* CHANGELOG 20.06.2023: change the three phenotype metrics to area, number, and radial velocity.
I think radius is not very well defined, whether it is from infection center or
from convex center, and which one, the average, min, max, etc. So better to use
area which shows the territory of the plaque/virus.

* Inconsistency for Radius information; in std output, the radius to center of 
infection is shown. In graph plots, the center to convex center is shown.

* num_cells changed to 3500 instead of previously 2750 in config files, based on
visually matching area and inf_count numbers.
(Before the recent change for num_cells: Default num_cells = 2750 number used in
configs are based on the density of cells computed on a single image with pixel
dimensions that could possibly be wrong. So be cautious about its correctness.
The calculations you can find at MacBookM1->workspace/Cellpose/01_analyze_results.ipynb)

* vmin and vmax of colorbar are based on 5 and 95 % of the data. If vmax is max, then the rest of the plot except for a few locations is almost in the very low range of colorbar range and not very clear. Needs better clarification, but most probably because molecule diffusion very very slow in the medium and the single point peak stays large during the diffusion for few steps.

* Assumption in code: x,y coords start from lower left. particle variable u also same, i.e. rows correspond with x and
columns with y, first row is x=0 and first column is y=0. Therefore for ploting u, need to do
plt.imshow(u.T, origin='lower')

* cells are points. So molecules are produced at a single location. They also do not collide, ...