## Bayesian State Space Approaches for Analyzing NFL Moneylines

### Table of Contents
1. [Motivation](#Motivation)
2. [Bayesian State Space Model](#BSSM)
3. [Logistic Regression Model](#logit)
4. [Data](#data)
5. [Results](#results)

### Motivation <a name="Motivation"></a>

* I love the NFL, but my favorite team, the Minnesota Vikings, does ***not*** always love me. The Vikings are eager to disappoint, so I sometimes feel like I need to other, healthier ways for staying close to the league. 
* In 2022, I completed my MS in statistics at the University of Minnesota. To do so, I had the opportunity to propose a new approach to modeling NFL moneylines. In addition to typical covariates informing a moneyline, my addition to the research was **conceiving of NFL moneyline updates as realizations of an underlying, unobservable state process.** In other words, we only see those moneyline updates over time, which are reflections of the underlying state process that describes the true game outcome.
* I'll show that thinking of NFL moneyline updates in this way ***entails a small amount of benefit*** over the sharps (sports books).

### Bayesian State Space Model <a name="BSSM"></a>

I assume moneyline updates from the sportsbooks are realizations of an underlying, unobservable process. Our primary objective in this section is to describe a method for drawing samples of this state process using a state space model and Gibbs sampler, which are used to construct a highest posterior density interval.

State space models correspond to a broad class of time series models characterized by two processes - a latent *state process*, $x_t$, and an *observation process*, $y_t$, each pertaining to a single home moneyline for a single NFL game. It is assumed that the state process is Markov (past and present are independent conditional on the present) and that observations are independent given states.[^fn] I will use the simplest state space model the AR(1), random walk model:

[^fn]: Shumway and Stoffer, *Time Series Analysis and Its Applications*, Fourth Edition, p. 289

$$x_t = x_{t-1}+w_t, w_t \sim \text{Normal}\left(0,\sigma_w^2\right) \text{ (State equation)}$$

$$y_t= x_t + v_t, v_t \sim \text{Normal}\left(0,\sigma_v^2\right) \text{ (Observation equation)}$$

From this parameterization, $y_t$ for $t\in \left[1,\dots,n\right]$ corresponds to the $t^{th}$ home moneyline update. The home moneyline update at time $t$, $y_t$, is a function of an underlying state AR(1) process at time $t$, $x_t$ for $t \in \left[0,\dots,n\right]$.

The most popular method for drawing samples of $x_t$ is Gibbs sampling - an MCMC procedure for sampling from intractable posterior distributions.[^fn2] As it pertains to the problem of sampling from the underlying sportsbook update process, I draw samples from the full conditional distribution of $\sigma_w^2$, $\sigma_v^2$ and $x_{0:n}$. In what follows, I will establish these full conditional distributions for each of these 2+n parameters, then combine into a Gibbs sampling algorithm that will be applied to each game in sample.

[^fn2]: Shumway and Stoffer, *Time Series Analysis and Its Applications*, Fourth Edition, p. 368

**In summary**: the observed moneyline update at time $t$, $y_t$, is treated as an outcome or realization of an underlying state process, which describes the actual game result. By directly modeling that state process, $x_t$, instead of the observed process, $y_t$, I want to see if I can approximate *how* a game moneyline is updated & exploit that in a wagering strategy.


### Logistic Regression Model <a name="logit"></a>


### Data <a name="data"></a>


### Results <a name="results"></a>
