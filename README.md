# Online–Offline Spillover

O2O is a Python package that quantifies mutual spillover between online and offline events, showing how activity in one arena can influence and be influenced by the other. Generating realistic synthetic data, it lets you prototype analyses before working with sensitive records. Using a bivariate Hawkes process, O2O delivers clear estimates of both the strength and direction of these reciprocal effects, whether you are examining social-media dynamics alongside real-world incidents or any other paired event streams.


<!--## Instalation

Install the package with:

pip install O2O-->


## Dependencies

To use O2O, make sure the following Python libraries are installed:

- pip install numpy pandas matplotlib stan nest_asyncio.
	
- The package was developed and tested with Python 3.9.5.

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


<!--Here, online activity (e.g., negative or threatening comments on social media) is indexed by 1, while offline activity (e.g., shootings) is indexed by 2. The parameter $\alpha$ captures the spillover effects between event types; for example, $\alpha_{12}$ represents the expected number of offline events triggered by a single online event. The parameter $\gamma_{ij}$ controls the decay rate of cross-excitation from events of type $j$ to type $i$, indicating how quickly the influence of past events fades over time. Finally, $\mu_{\text{user}} = [\mu_1^{\text{user}}, \mu_2^{\text{user}}]$ denotes the baseline intensities for online and offline activities for each user, respectively.-->

## Aknowledgement

This package is based on:

John Leverso, Youness Diouane, George Mohler, "Measuring Online–Offline Spillover of Gang Violence Using Bivariate Hawkes Processes", 
Journal of quantitative criminology, 2025.






