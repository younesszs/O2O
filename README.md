# Online $\leftright$ Offline Spillover

O2O is a Python package that estimates the magnitude and direction of spillover effects between online and offline activities. One key application is analyzing how social media behavior, such as threatening or hostile comments made by gang members on platforms like Facebook, may contribute to offline incidents like shootings. By modeling these interactions using a bivariate Hawkes process. 


## Instalation
Instal the package using pip install O2O

API documentation is provided in O2O_API_documentation.pdf



## Usage

After installation, the user specifies:

- The number of users
- The time period of interest

The package then performs the following:

### 1. **Generate synthetic timestamp data**

- Simulates timestamps for both online and offline user events.

### 2. **Estimate spillover effects**

- Calculates the influence of one event type (online/offline) on the other.
- Provides:
  - 95% confidence intervals for spillover effect estimates.
  - Decay rates and their 95% confidence intervals.
  - The percentage of events caused by the other type.

### 3. **Summarize user activity**

- Computes the total number of online and offline events per user.
- Plots the coupled online–offline intensity over time for each user.

# Model

The model in a bivariate Hawkes process that can be described by the conditional intensities

$$
\lambda_1^{\text{user}} = \mu_1^{\text{user}} + \sum\limits_{k:t>t_k^1}^{N_\text{user}^1} \alpha_{11}\beta_{11} e^{-\beta_{11}(t-t_k^1)} + \sum\limits_{k:t>t_k^2}^{N^2_\text{user}} \alpha_{12}\beta_{12} e^{-\beta_{12}(t-t_k^2)}
$$
$$
\lambda_2^{\text{user}} = \mu_2^{\text{user}} + \sum\limits_{k:t>t_k^1}^{N^1_\text{user}} \alpha_{21}\beta_{21} e^{-\beta_{21}(t-t_k^1)} + \sum\limits_{k:t>t_k^2}^{N^2_\text{user}} \alpha_{22}\beta_{22} e^{-\beta_{22}(t-t_k^2)},
$$


Here online activity e.g., negative comments of threats on social media are indexed by 1 and offline activities e.g., shootings are marked by 2. The parameter $\alpha$ gives us the spillover effect, for example $\alplha_{12}$ accounts for the expected number of offline activities caused by an initial social media negative post, whereas $\beta_{ij}$ determines the decay rate of cross-excitation from node $j$ to node $i$. Lastly, $\mu_{\text{user}} = [mu_1^{\text{user}}, mu_2^{\text{user}}]$ are the baseline rates for online and offline activities, respectively. 

### Aknowledgement

This package is based on [1]

[1]: John Leverso, Youness Diouane, George Mohler, Measuring Online–Offline Spillover of Gang Violence Using Bivariate Hawkes Processes.
Journal of quantitative criminology, 2025.





