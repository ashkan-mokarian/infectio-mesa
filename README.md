# Infectio

Agent-based modeling for Virological Plaque using Mesa library.

# Personal notes

* Can we avoid discretization of particle scalar fields and work completely with
well defined function. For example solve the diffusion with it, and also for
gradient computations be able to use [this python package: numdifftools](https://numdifftools.readthedocs.io/en/latest/index.html).

* Assumption in code: x,y coords start from lower left. particle variable u also same, i.e. rows correspond with x and
columns with y, first row is x=0 and first column is y=0. Therefore for ploting u, need to do
plt.imshow(u.T, origin='lower')

* Now, molecule released at int(x, y). Could also release with sub-pixel precision but not implemented.