<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
  
# Chapter 2

## Multi-Armed Brandits

### A k-armed Bandit Problem

问题：k actions，每个action有一个对应的reward (has a stationary probability distribution).

Objective: maximize the expected total reward

**Value**: the expected or mean reward for a given action

**$A_t$**: action at time t

$R_t$: reward for action at time t

the value for action a: $q_*(a) = E[R_t\|A_t = a]$

**Estimation value of action a at time t**: $Q_t(a)$

would like $Q_t(a)$ to be close to $q_*(a)$



**Methods for taking actions**: 

1. Exploiting: at any time step taking the action whose estimated value is greatest
2. Exploring: select one of the nongryeedy actions (enables you to improve your estimate of the nongreedy action's value)

如果往后time frame足够长，可以尝试exploring，虽然短期expected reward可能会小，但是可能会找到potential 更好的bandit，然后exploit。可能会有long term更高的expected reward。

### Action-value Methods

simple average:

$Q_t(a)=\frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}}$

By the law of large numbers, $Q_t(a)$ converges to $q_*(a)$

**Greedy action selection methods:** $A_t=argmax_a Q_t(a)$

**$\epsilon$-greedy**: with probability $\epsilon$, select randomly from among all the actions

*Advantage*: in the limits as the number of steps increases, every action will be sampled an infinite number of times. Ensures $Q_t(a)$ converge to $q_*(a)$. (the probability of selecting the optimal action converges to greater than $1-\epsilon$, plus the probability that selecting the greedy action when randomly selecting)

if the reward variance had been larger, it takes more exploration to find the optimal action. 

if the bandit task were non stationary, that is the true values of the actions changed over time. also need exploration.

### Incremental Implementation

implementations of the the action-value methods: constant memory and constant per-time-step computation

$R_i$: the reward received after the ith selection of this action.

$Q_n$: denote the estimate of its action value after it has been selected $n-1$ times.

$$
Q_n = \frac{R_1+R_2+\cdots+R_{n-1}}{n-1}
$$

$$
Q_{n+1}=Q_n+\frac{1}{n}(R_n-Q_n)
$$

```python
Inialize, for a=1 to k:
  Q(a) <- 0
  N(a) <- 0
  
  while True:
    A = argmax Q(a) w.p. 1-eps; a random action w.p. eps
    R <- bandit(A)
    N(A) <- N(A)+1
    Q(A) <- Q(A)+(R-Q(A))/N(A)
```

### Tracking a Nonstationary Problem

In non-stationary cases, it makes sense to give more weight to recent rewards than to long-past rewards. 

$$
\begin{align}Q_{n+1} &= Q_n+\alpha(R_n-Q_n)\\&=\alpha R_n + (1-\alpha)Q_n\\&=\alpha R_n + (1-\alpha)[\alpha R_{n-1}+(1-\alpha Q_{n-1})]\\&=(1-\alpha)^n Q_1 + \sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i\end{align}
$$

$(1-\alpha)^n+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}=1$, **exponential recency-weighted average**

**Conditions required to assure convergence with prob 1**:

$\alpha_n(a)$ the step-size parameter used to process the reward received after the n-th selection of action a.

$\sum_{n=1}^{\infty} \alpha_n(a)=\infty$: guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations.

$\sum_{n=1}^{\infty}\alpha^2(a)<\infty$: gaurantees that eventually the steps become small enough to assure convergence.

### Optimistic Initial Values

previous methods depend to some extent on the initial action-value estimates, $Q_1(a)$. Therefore, these methods are biased by their initial estimates. 

**Downside**: the initial estimates become a set of parameters that must be picked by the user.

**Advantage**: 

1. supply some prior knowledge about hte level of rewards can be expected

2. Encourage exploration: stationary case 将初始值设置为比distribution均值高的。The result is that all actions are tried several times before the value estimates converge. 

    Not useful for nonstationary distribution: the drive for exploration is inherently temporary.

### Upper-Confidence-Bound Action Selection

$\epsilon$-greedy: indiscriminately select the non-greedy actions, with no preference for those that are nearly greedy or particularly uncertain.

$$
A_t = argmax_a Q_t(a)+c\sqrt{\frac{In(t)}{N_t(a)}}
$$

the latter term is a measure of the uncertainty or variance:

1. action a is taken, $N_t(a)$ increase, uncertainty decrease
2. Action a is not taken at time t, t increase, $N_t(a)$ stay same, uncertainty increase. the square root make sure that the increases get smaller over time.

Note: not practical for nonstationary and function approximation methods

### Gradient Bandit Algorithms

$H_t(a)$: a numerical preference for each action a at time t; the larger the preference, the more often that action is taken. 

$$
\mathbb{P}(A_t=a) = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}} = \pi_t(a)
$$

$\pi_t(a)$: the probability of taking action a at time t. $H_1(a)=0$

$$
H_{t+1}(a) = H_t(a)+\alpha\frac{\partial\mathbb{E}[R_t]}{\partial H_t(a)}\\\mathbb{E}[R_t] = \sum_x \pi_t(x)q_*(x)\\
$$

$$
H_{t+1}(A_t) = H_t(A_t)+\alpha(R_t-\bar{R}_t)(\mathbf{1}_{\{A_t=a\}}-\pi_t(A_t))
$$

$\bar{R}_t$ is the average of all the rewards up through and including time t (computed incremental methods). serve as a **baseline**. $\sum_x \frac{\partial\pi_t(x)}{\partial H_t(a)} = 0$ So we can include the baseline.

1. If the reward is higher than the baseline, then the prob of taking $A_t$ in the future is increased.
2. If the reward is below baseline, then prob is decreased.

### Associative Search

**Policy**:a mapping from situations to the actions that are best in those situations

**Associative search task**: it involves both trial- and-error learning to *search* for the best actions, and *association* of these actions with the situations in which they are best. Associative search tasks are often now called **contextual bandits** in the literature. 
### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
