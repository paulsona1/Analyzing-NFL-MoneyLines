## Bayesian State Space Approaches for Analyzing NFL Moneylines

### Table of Contents
1. [Motivation](#Motivation)
2. [Bayesian State Space Model](#BSSM)
    * [Full conditional distribution of process variance](#proc_var)
    * [Full conditional distribution of state process](#proc_state)
4. [Logistic Regression Model](#logit)
5. [Data](#data)
6. [Results](#results)

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

#### Full conditional distribution of process variances $\sigma_w^2$, $\sigma_v^2$ <a name="proc_var"></a>

These are not ***so*** bad, if we take advantage the posterior distribution for Normal likelihood & Inverse Gamma prior. State ($\sigma_w^2$) and observation ($\sigma_v^2$) variances will each rely on normal-inverse gamma conjugacy. Assuming normal likelihood for sportsbook updates and inverse gamma prior for $\sigma_w^2$ and $\sigma_v^2$,[^fn3] the posterior distribution for $\sigma_w^2$ and $\sigma_v^2$ is well known.

[^fn3]: See for example Glickman, M. E., & Stern, H. S. (1998). A State-Space Model for National Football League Scores. Journal of the American Statistical Association, 93(441), 25–35. https://doi.org/10.2307/2669599

For $\sigma_v^2$, we will assume the inverse gamma prior with mean equal to the sample variance of sportbook updates across all games in the prior week, $s_{v,\text{prev}}^2$ and scale parameter set at 70. Thus:
$$\sigma_v^2 \sim \text{IG}\left(\alpha_v=1+\beta_v/s_{v,\text{prev}}^2,\beta_v=70\right)$$
Implying:
$$\sigma_v^2|y_{1:n} \sim \text{IG}\left(\frac{1}{2}\left(\alpha_v+n\right),\frac{1}{2}\left(\beta_v+\displaystyle\sum_{i=1}^n\left(x_t-y_t\right)^2\right)\right)$$
For $\sigma_w^2$, we assume the state process entails slightly less variance. For this reason, we will use the scale parameter $\beta_w=(4/3)\times\beta_v$, but retain quality of the $\sigma_v^2$ prior being centered at $s^2_{v,\text{prev}}$:
$$\sigma_w^2 \sim \text{IG}\left(\alpha_w=1+\beta_w/s^2_{v,\text{prev}},\beta_w=(4/3)\times\beta_v\right)$$
Implying:
$$\sigma_w^2|x_{0:n} \sim \text{IG}\left(\frac{1}{2}\left(\alpha_w+n\right),\frac{1}{2}\left(\beta_w+\displaystyle\sum_{i=1}^n\left(x_t-x_{t-1}\right)^2\right)\right)$$

#### Full conditional distribution of state process <a name="proc_state"></a>

Ok, this part sucks. The most challenging step in this entire process pertains to obtaining the filtering densities at each time $t$. We will use the Gaussianity of the state process and the Markov assumption to derive these densities. We are interested in $p\left(x_t|y_{1:t}\right)$. Because the state process is Markov:
$$p\left(x_{0:n}|y_{1:n}\right) = p\left(x_n|y_{1:n}\right) \times p\left(x_{n-1}|x_n,y_{1:n-1}\right) \times \dots \times p\left(x_t|x_{t+1},y_{1:t}\right) \dots \times p\left(x_0|x_1\right)$$
$$=p\left(x_0|x_1\right) \times \prod_{t=1}^{n} p\left(x_t|x_{t+1},y_{1:t}\right)$$
Where:
$$p\left(x_t|x_{t+1},y_{1:t}\right)\propto p\left(x_{t+1}|x_t\right)\times p\left(x_t|y_{1:t}\right)$$
By the state equation, $p\left(x_{t+1}|x_t\right)$ is $\text{Normal}\left(x_t,\sigma_w^{2}\right)$. So it remains to find density $p\left(x_t|y_{1:t}\right)$.[^fn4]

[^fn4]:This is the Kalman update equation.

$$p\left(x_t|y_{1:t}\right) = p\left(x_t|y_t,y_{1:t-1}\right) \propto p\left(y_t|x_t\right)\times p\left(x_t|y_{1:t-1}\right)$$
By the observation equation, $p\left(y_t|x_t\right)$ is $\text{Normal}\left(x_t,\sigma_v^2\right)$. Now, it remains to find density $p\left(x_t|y_{1:t-1}\right)$[^fn5]

[^fn5]:This is the Kalman prediction equation.

Using the Chapman-Kolmogorov equation and the normality of the state process:[^fn6]

[^fn6]:Shumway and Stoffer, *Time Series Analysis and Its Applications*, Fourth Edition, p. 297

$$p\left(x_t|y_{1:t-1}\right) =\displaystyle\int p\left(x_t,x_{t-1}|y_{1:t-1}\right) dx_{t-1} = \displaystyle\int p\left(x_t|x_{t-1}\right)\times p\left(x_{t-1}|y_{1:t-1}\right)d x_{t-1}$$
By the state equation, $p\left(x_t|x_{t-1}\right)$ is $\text{Normal}\left(x_{t-1},\sigma_w^2\right)$. Using the notation of Blight (1974), let $\left(x;\mu,\sigma^2\right)$ denote a normal density with mean $\mu$ and variance $\sigma^2$.[^fn7]

[^fn7]: Blight BJN (1974), *Recursive solutions for the estimation of stochastic parameter*. Journal of American Statistical Association 69:477-481] 

Then:
$$p\left(x_t|y_{1:t-1}\right) = \displaystyle\int \left(x_t;x_{t-1},\sigma_w^2\right)\times \left(x_{t-1}; x_{t-1}^{t-1},P_{t-1}^{t-1}\right) d_{x_{t-1}} = \displaystyle\int \left(x_{t-1};x_t,\sigma_w^2\right)\times \left(x_{t-1}; x_{t-1}^{t-1},P_{t-1}^{t-1}\right) d_{x_{t-1}}$$
where $x_{t-1}^{t-1}$ and $P_{t-1}^{t-1}$ follow from the Kalman filter.[^fn8] Here, we are using the fact that $E\left[x_{t-1}|y_{1:t-1}\right]=x_{t-1}^{t-1}$ and $\text{Var}\left[x_{t-1}|y_{1:t-1}\right]=P_{t-1}^{t-1}$.

[^fn8]: See **How to Kalman Fil1ter**

Completing the square[^fn6]:

[^fn6]:See **How to Complete the Square**

$$p\left(x_t|y_{1:t-1}\right) = \displaystyle\int \left(x_{t-1};\frac{x_{t-1}^{t-1}\sigma_w^2 + x_t P_{t-1}^{t-1}}{\sigma_w^2+P_{t-1}^{t-1}},\frac{\sigma_w^2 P_{t-1}^{t-1}}{\sigma_w^2+P_{t-1}^{t-1}}\right)\times\left(x_t;x_{t-1}^{t-1},P_{t-1}^{t-1}+\sigma_w^2\right) dx_{t-1}$$
which evaluates to $\left(x_t;x_{t-1}^{t-1},P_{t-1}^{t-1}+\sigma_w^2\right)$, so $p\left(x_t|y_{1:t-1}\right)$ is $\text{Normal}\left(x_{t-1}^{t-1},P_{t-1}^{t-1}+\sigma_w^2\right)$. Therefore:
$$p\left(x_t|y_t,y_{1:t-1}\right)=p\left(x_t|y_{1:t}\right) \propto p\left(y_t|x_t\right)\times p\left(x_t|y_{1:t-1}\right) = \left(y_t;x_t,\sigma_v^2\right)\times\left(x_t;x_{t-1}^{t-1},P_{t-1}^{t-1}+\sigma_w^2\right)$$
Using the Gaussianity:
$$p\left(x_t|y_t,y_{1:t-1}\right) \propto \left(x_t;y_t,\sigma_v^2\right)\times\left(x_t;x_{t-1}^{t-1},P_{t-1}^{t-1}+\sigma_w^2\right)$$
Completing the square again:
$$p\left(x_t|y_t,y_{1:t-1}\right) \propto \left(x_t;\frac{y_t\left(P_{t-1}^{t-1}+\sigma_w^2\right)+x_{t-1}^{t-1}\sigma_v^2}{P_{t-1}^{t-1}+\sigma_w^2+\sigma_v^2},\frac{\sigma_v^2\left(P_{t-1}^{t-1}+\sigma_w^2\right)}{P_{t-1}^{t-1}+\sigma_w^2+\sigma_v^2}\right)$$
$$= \left(x_t; x_{t-1}^{t-1} + K_t\left(y_t-x_{t-1}^{t-1}\right),\left(1-K_t\right)\times \left(P_{t-1}^{t-1}+\sigma_w^2\right)\right)$$
where $K_t=\frac{P_{t-1}^{t-1}+\sigma_w^2}{P_{t-1}^{t-1}+\sigma_w^2+\sigma_v^2}$ is the Kalman gain. 

So the forward filtering density at time $t$ is:

$$p\left(x_t|y_{1:t}\right) \propto \text{Normal}\left(x_{t-1}^{t-1}+K_t\left(y_t-x_{t-1}^{t-1}, \left(1-K_t\right)\times\left(P^{t-1}_{t-1}+\sigma^2_w\right)\right)\right)$$



Therefore, by initializing $x_{0}^{0}=\mu_0$ and $P_0^0=\sigma_0$, we have each of the $n+1$ forward filtering densities (from $0$ to $n$), which are special cases of the Kalman filter. From the initialization, it can be seen that $x_n^n=x_{n-1}^{n-1}+K_n\left(y_n-x_{n-1}^{n-1}\right)$ and $P_n^n=\left(1-K_n\right)\times\left(P_{n-1}^{n-1}+\sigma_w^2\right)$. Drawing a sample from the normal distribution having mean $x^n_n$ and variance $P_n^n$ gives us $x_n$.^[Notably, our analysis will rely strictly on $x_n$ and the highest posterior density around it this sample. For completeness, we will show the entire back sampling algorithm for drawing samples of $x_{n-1}:x_0$.]

We will now use that sample at the $n^{th}$ moneyline update as part of the Kalman smoother to back sample $x_t$ for $t=n-1,\dots,0$. Here, we are interested in $p\left(x_{t}|x_{t+1}\right)$. However, because $x_t$ and $x_{t+1}$ are jointly normal:^[Shumway and Stoffer, *Time Series Analysis and Its Applications*, Fourth Edition, p. 370]
$$
\begin{pmatrix}
x_t \\
x_{t+1}
\end{pmatrix}| y_{1:t} \sim \text{Normal}\left(\begin{pmatrix}
x_{t}^{t} \\
x_{t+1}^{t} = x_t^t
\end{pmatrix},\begin{pmatrix}
P_t^t & P_t^t \\P_t^t & P_{t+1}^t=P_{t}^{t}+ \sigma_w^2
\end{pmatrix}\right)
$$
By **Appendix B.2**, the conditional distribution $p\left(x_{t}|x_{t+1}\right)$ is: 
$$
\text{Normal}\left(x_t^t+J_t\left(x_{t+1}-x_{t}^{t}\right),P_t^t-J_t^2\left(P_t^t + \sigma_w^2\right)\right)
$$
where $J_t =P_t^t\left(P_t^t +\sigma_w^2\right)^{-1}$.

### Logistic Regression Model <a name="logit"></a>


### Data <a name="data"></a>


### Results <a name="results"></a>
