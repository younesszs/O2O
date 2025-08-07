# Online–Offline Spillover   <p align="right">
  <a href="https://github.com/younesszs/O2O">
    <img src="assets/O2O_QR.png" alt="QR Code" width="150">
  </a>
</p>

O2O is a Python package that quantifies mutual spillover between online and offline events, showing how activity in one arena can influence and be influenced by the other. Generating realistic synthetic data, it lets you prototype analyses before working with sensitive records. Using a bivariate Hawkes process, O2O delivers clear estimates of both the strength and direction of these reciprocal effects, whether you are examining social-media dynamics alongside real-world incidents or any other paired event streams.


## Installation

- I highly recommend installing Python and the required dependencies in a virtual environment such as Conda. On Unix-based systems (Linux, macOS), Python is often tied to critical system functions, so modifying the system-wide Python installation can be risky. To install Conda:

- Visit the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.
- Follow the installation instructions.
- Once installed, create and activate a dedicated environment

```bash
conda create -n o2o_env python=3.11
conda activate o2o_env
```
- Once your environment is activated, install the required dependencies:

```bash
pip install numpy pandas matplotlib nest_asyncio
```
Then install CmdStan via conda. 

```bash
conda install -c conda-forge cmdstan cmdstanpy
```

This is to avoid manual build issues and ensures CmdStan is precompiled and avoids needing `make` or compiler setup during install. 

- Once you install all the dependencies, you can install the package using 

```bash
pip install o2o-process
```
- API documentation is provided in O2O_API_documentation.pdf in the docs directory.

## Usage

After installing the package, you can run its demo using the command:

```bash
o2o-demo
```
where you will be prompted to specify:

- The number of users
- The time window of interest (in days)

Note: To avoid generating empty synthetic data, use at least 5 users and a minimum of 20 days (T = 20). Smaller values may not yield meaningful or sufficient activity data.

You can also run the demo from a Jupyter Notebook (`demo.ipynb`). To access it, clone the GitHub repository:

```bash
git clone https://github.com/younesszs/O2O.git
```

### Note for Windows users
Stan models require a C++ compiler and the `make` build tool, neither of which are included by default on Windows. Installing and configuring these tools (e.g., `g++`, `make`, and a Unix-like shell) can be complex and error-prone due to differences between Unix-based systems and Windows. In fact, Windows' built-in Command Prompt (cmd.exe) and PowerShell do not support Unix commands by default. But tools like `Stan`, `make`, and `cmdstanpy` often assume they’re running in a Unix-style shell. This is why I highly recomment for Windows users to use Google Colab, a free, browser-based Python environment that runs on Linux and includes all necessary tools to compile and run Stan models out of the box. To do that follow the steps:

- Go to https://colab.research.google.com and open a new Python notebook and run

```bash
!pip install numpy pandas matplotlib nest_asyncio
```
- Then install Stan using
```bash
import cmdstanpy
cmdstanpy.install_cmdstan()
```
- Install the package:
```bash
!pip install o2o-process
```

- Finally, you can run the O2O demo through

```bash
from o2o.demo import main
main
```
- You can also use the included Jupyter Notebook  `demo.ipynb` directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/younesszs/O2O/blob/main/demo.ipynb)

Note: Make sure to uncomment and run the first cell before executing the rest of the notebook. It installs all required dependencies.




The package performs the following:


### 1. **Generates synthetic timestamp data**

- Simulates timestamps for both online and offline user events.

### 2. **Estimates spillover effects**

- Calculates the influence of one event type (online/offline) on the other and its corresponding 95% confidence interval.
- Decay rates and their 95% confidence intervals.
- The percentage of events caused by the other type.
- Estimates for the baseline intensities online and offline for each user as well as their corresponding 95% confidence interval.

### 3. **Summarizes user activity**

- Computes the total number of online and offline events per user.
- Plots the coupled online–offline intensity over time for each user.

## Model

The model is a bivariate Hawkes process that can be described by the conditional intensities

$$
\lambda_1^{\text{user}} = \mu_1^{\text{user}} + \sum\limits_{k:t>t_k^1}^{N_\text{user}^1} \alpha_{11}\gamma_{11} e^{-\gamma_{11}(t-t_k^1)} + \sum\limits_{k:t>t_k^2}^{N^2_\text{user}} \alpha_{12}\gamma_{12} e^{-\gamma_{12}(t-t_k^2)}
$$
$$
\lambda_2^{\text{user}} = \mu_2^{\text{user}} + \sum\limits_{k:t>t_k^1}^{N^1_\text{user}} \alpha_{21}\gamma_{21} e^{-\gamma_{21}(t-t_k^1)} + \sum\limits_{k:t>t_k^2}^{N^2_\text{user}} \alpha_{22}\gamma_{22} e^{-\gamma_{22}(t-t_k^2)},
$$

where:

* Online activity (e.g., hostile posts) is indexed by 1.
* Offline activity (e.g., shootings) is indexed by 2.
* $\alpha_{ij}$: expected number of type-i events triggered by an initial type-j event.
* $\gamma_{ij}$: decay rate of influence from type $j$ to type $i$
* $\mu^{\text{user}} = [\mu_1, \mu_2]$: baseline intensities per user



## Acknowledgement

- This work was supported in part by AFOSR grant FA9550-22-1-0380.

- This package is based on:

John Leverso, Youness Diouane, George Mohler, "Measuring Online–Offline Spillover of Gang Violence Using Bivariate Hawkes Processes", 
Journal of quantitative criminology, 2025.

## Feedback and contributions
I am always open to improving this software! 
Feel free to open an issue, submit a pull request, or suggest enhancements.

If you find this project useful, feel free to star it to help others discover it.

Thanks for using O2O!









