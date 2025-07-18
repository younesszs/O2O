# Online–Offline Spillover

O2O is a Python package that quantifies mutual spillover between online and offline events, showing how activity in one arena can influence and be influenced by the other. Generating realistic synthetic data, it lets you prototype analyses before working with sensitive records. Using a bivariate Hawkes process, O2O delivers clear estimates of both the strength and direction of these reciprocal effects, whether you are examining social-media dynamics alongside real-world incidents or any other paired event streams.


<!--## Instalation

Install the package with:

pip install O2O-->


## Dependencies

To use O2O, make sure the following Python libraries are installed:

- pip install numpy pandas matplotlib pystan nest_asyncio.
	
- The package was developed and tested with Python 3.9.5. I highly recommend installing Python and the required dependencies in a virtual environment such as Conda. On Unix-based systems (Linux, macOS), Python is often tied to critical system functions, so modifying the system-wide Python installation can be risky. To install Conda:

- Visit the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.
- Follow the installation instructions.
- Once installed, create and activate a dedicated environment

```bash
conda create -n o2o_env python=3.9
conda activate o2o_env
```
Once your environment is activated, install the required dependencies.

- API documentation is provided in O2O_API_documentation.pdf in the docs directory.

## Usage

You can use O2O in two ways:

- From a Jupyter Notebook (`demo.ipynb`)
- From the terminal using the Python script (`demo.py`)

When using `demo.py`, you will be prompted to specify:

- The number of users
- The time window of interest (in days)

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



## Aknowledgement

This package is based on:

John Leverso, Youness Diouane, George Mohler, "Measuring Online–Offline Spillover of Gang Violence Using Bivariate Hawkes Processes", 
Journal of quantitative criminology, 2025.






