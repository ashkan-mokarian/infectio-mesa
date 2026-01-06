# infectio: Agent-Based Model of VACV Infection

**infectio** is a computational model built using the [Mesa](https://github.com/projectmesa/mesa) framework to simulate the spread of Vaccinia Virus (VACV) in an epithelial cell monolayer.

The model simulates cell-to-cell infection and cell motility driven by chemotaxis. It works by solving diffusion equations for signaling molecules secreted by infected cells, creating gradients that guide the migration of cells.

## 📋 Table of Contents
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dual Diffusion Mechanics](#-dual-diffusion-mechanics)
- [Configuration](#-configuration)
- [Output & Visualization](#-output--visualization)
- [Project Structure](#-project-structure)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://codebase.helmholtz.cloud/casus/yakimovich/infectio](https://codebase.helmholtz.cloud/casus/yakimovich/infectio)
   cd infectio
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   conda create -n infectio python=3.11
   conda activate infectio
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

The main entry point for the simulation is `infectio/models/vacv/run.py`. You can run the simulation by passing a configuration file.

**Run a baseline simulation (Double Deletion):**
This runs the model with diffusion disabled (Random Walk only). Use this to verify the basic cellular automata are working.
```bash
python infectio/models/vacv/run.py -c examples/vacv-epithelial/dVGFdF11_default_config.ini
```

**Run a debug simulation**
This runs a configuration designed to test the diffusion solver and signal visualization.
```bash
python infectio/models/vacv/run.py -c examples/vacv-epithelial/debug_config.ini --run_gui
```

**Run Wild Type (WR) with Dual Diffusion**

```bash
python infectio/models/vacv/run.py -c examples/vacv-epithelial/WR_config.ini
```

## 🧬 Dual Diffusion Mechanics

This model simulates the chemotactic movement of infected cells. This is driven by two distinct pathways:

1.  **VGF**
2.  **F11**

### Logic
* **Secretion:** Infected cells secrete VGF and F11 particles into the grid.
* **Diffusion:** These particles diffuse over time based on their respective alpha (diffusion constant) values.
* **Chemotaxis:** Uninfected cells sense the local gradient of both molecules. The final movement vector is the sum of the displacement vectors derived from both the VGF and F11 gradients.

### Supported Scenarios
* **Wild Type (WR):** Both `enable_vgf` and `enable_f11` are **True**.
* **∆VGF:** Only `enable_f11` is **True**.
* **∆F11:** Only `enable_vgf` is **True**.
* **∆VGF∆F11 (Double Deletion):** Both are **False** (cells move via random walk only).

## ⚙️ Configuration

The model is configurable via `.ini` files or Command Line Arguments. Arguments passed via CLI override those in the config file.

### Complete Parameter Reference

The configuration is divided into sections corresponding to the `.ini` file structure.

#### 1. General Settings `[general]`
Controls simulation execution, output paths, and logging.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `save_root` | Path | Root directory for saving outputs (e.g., `output/debug/WR`). |
| `save_name` | String | Sub-folder name for the run. Auto-generated if left empty. |
| `run_gui` | Bool | `True` for live visualization, `False` for faster headless/cluster runs. |
| `savesnapshots` | Bool | If `True`, saves `.png` images of the visualization. Useful for debugging. |
| `plot_verbosity` | Int | Level of detail for real-time plots (1=Low, 3=High). |
| `reference_file` | Path | Path to experimental CSV data for real-time metric comparison. |

#### 2. Model Dimensions & Time `[model]`
Defines the spatiotemporal resolution of the simulation.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `n_sim_steps` | Int | Total number of simulation steps to run. |
| `time_per_step` | Int | Real-world time represented by one step (in seconds). |
| `pixel_length` | Float | Physical size of one grid pixel ($\mu m$). |
| `width`, `height` | Int | Grid dimensions in pixels. |

#### 3. Cell Biology `[cell]`
Parameters governing cell density, random movement, and infection dynamics.

**Density & Placement**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `cell_density_mean` | Float | Target density of cells on the grid. |
| `cell_density_std` | Float | Standard deviation for density generation. |
| `initial_random_placement_method` | String | Method for seeding cells ("einstein" for clustered noise and "poisson" for uniformaly random placement). |
| `initial_random_placement_einstein_factor`| Float | Scales the random displacement from a perfect grid. Multiplies the average inter-cell spacing to set the noise standard deviation. |

**Motility & Infection**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `randomwalk_speed` | Float | Maximum Brownian motion speed for all cells ($\mu m/min$). The actual step size is sampled uniformly from [0, max]. |
| `c2c_radius_search` | Float | Radius to search for susceptible neighbors ($\mu m$). |
| `c2c_sigmoid_k` | Float | Steepness of the infection probability curve. |
| `c2c_sigmoid_t0` | Float | Time offset for the infection curve (Hours). |
| `c2c_sigmoid_tmid` | Float | Time (Hours) at which infection probability is 50%. |
| `first_cell_lag` | Float | Delay (Hours) before the initial seed cell becomes active. |

#### 4. Dual Diffusion `[diffusion]`
Controls the secretion, diffusion, and chemotaxis of VGF and F11.

**VGF Pathway**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enable_vgf` | Bool | Toggle VGF mechanism on/off. |
| `vgf_alpha` | Float | Diffusion constant $D$ ($\mu m^2/s$). |
| `vgf_diff_steps` | Int | Number of numerical solver iterations per simulation step. |
| `vgf_gradient_speed` | Float | Chemotactic speed toward VGF ($\mu m/min$). |
| `vgf_produce_max` | Float | Max secretion rate (arbitrary units). |
| `vgf_produce_t1` | Float | Time (Hours) post-infection when secretion starts. |
| `vgf_produce_t2` | Float | Time (Hours) post-infection when secretion stops. |

**F11 Pathway**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enable_f11` | Bool | Toggle F11 mechanism on/off. |
| `f11_alpha` | Float | Diffusion constant $D$ ($\mu m^2/s$). |
| `f11_diff_steps` | Int | Solver iterations per step. |
| `f11_gradient_speed` | Float | Chemotactic speed toward F11 ($\mu m/min$). |
| `f11_produce_max` | Float | Max secretion rate. |
| `f11_produce_t1` | Float | Start time (Hours) for F11 secretion. |
| `f11_produce_t2` | Float | Stop time (Hours) for F11 secretion. |

**Sensing Noise**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `gradient_direction_noise_max` | Float | Maximum rotational noise added to the movement vector (Radians). |

### ⚠️ Numerical Stability Note

The diffusion solver uses an explicit Forward Euler method. To prevent the simulation from diverging (exploding values), the number of diffusion steps per simulation step (`vgf_diff_steps`, `f11_diff_steps`) must satisfy the **Von Neumann stability condition**:

$$N \ge \frac{4 \cdot D \cdot \Delta t_{step}}{\Delta x^2}$$

Where:
* $N$: Number of diffusion steps (`*_diff_steps`).
* $D$: Diffusion constant (`*_alpha`).
* $\Delta t_{step}$: Time per simulation step (`time_per_step`).
* $\Delta x$: Grid pixel size (`pixel_length`).

**Rule of Thumb:** If you increase the diffusion constant (`alpha`) or the time step, you must proportionally increase `diff_steps` to maintain stability.

## 📊 Output & Visualization

### GUI Mode
If `run_gui = True`, a window will appear showing:
1.  **Cell Position Map:** Blue (Susceptible) and Green (Infected) cells.
2.  **Diffusion Heatmap:** Visualizes the combined concentration of VGF and F11.
3.  **Metrics:** Real-time plots of plaque area, radial velocity, and infection counts.

### Data Output
Results are saved to `output/<experiment_name>/`:
* `pos.csv`: Frame-by-frame positions of every cell.
* `metric.csv`: Time-series data of global metrics (Infected Count, Area, Velocity).
* `params.json`: A snapshot of the configuration used for that run.
* `plots/`: Snapshots of the simulation state (if enabled).

## 📂 Project Structure

```text
infectio-mesa/
├── examples/                   # Pre-defined configuration files
│   └── vacv-epithelial/        # Configs for WR, dVGF, dF11, etc.
├── infectio/
│   ├── models/vacv/            # Core model logic
│   │   ├── run.py              # Main entry script
│   │   ├── model.py            # Mesa Model class
│   │   ├── cell.py             # Cell Agent logic
│   │   └── default_config.ini  # Base configuration
│   ├── visualization/          # Matplotlib GUI logic
│   ├── particle.py             # Diffusion grid solver
│   └── reporters.py            # Data collection logic
├── notebooks/                  # Jupyter notebooks for post-analysis
├── scripts/                    # Utilities for HPC/Slurm execution
└── output/                     # Default save location for runs
```

## 🧪 Experimental Reference Data

The `examples/vacv-epithelial/` directory includes CSV files (e.g., `reference_metrics_...csv`) containing quantitative metrics extracted from time-lapse microscopy videos of VACV infection *in vitro*.

These files serve as "Ground Truth" baselines. When a simulation is run with the `reference_file` parameter pointing to one of these datasets, the GUI will overlay the experimental data (dotted lines) onto the real-time simulation plots (solid lines) for direct validation.

**Included Datasets:**
* `...WR_handpicked.csv`: **Wild Type** (VGF + F11 active).
* `...dVGF_handpicked.csv`: **∆VGF** (F11 only).
* `...dF11_handpicked.csv`: **∆F11** (VGF only).
* `...dVGFdF11_handpicked.csv`: **Double Deletion** (Baseline random walk).

*Note: These metrics were generated using a separate image analysis pipeline on raw experimental footage.*

## 🖥️ Running on Clusters (HPC)

For large-scale parameter sweeps, use the scripts provided in `scripts/`.
1.  Generate a list of parameters using `generate_param_list_file_for_array.py`.
2.  Submit the job array using `submit_param_array_from_listfile.sh`.

Ensure `run_gui = False` in your config when running on clusters.