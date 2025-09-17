# Statistics and Probability for Machine Learning

Statistics and probability form the theoretical foundation of machine learning. Understanding these concepts is crucial for grasping how algorithms learn from data, make predictions, and quantify uncertainty.

## Table of Contents

1. [Descriptive Statistics](#descriptive-statistics)
2. [Probability Theory](#probability-theory)
3. [Probability Distributions](#probability-distributions)
4. [Inferential Statistics](#inferential-statistics)
5. [Bayes' Theorem](#bayes-theorem)
6. [Hypothesis Testing](#hypothesis-testing)
7. [Practice Problems](#practice-problems)

## Descriptive Statistics

Descriptive statistics help us understand and summarize datasets.

### Measures of Central Tendency
- **Mean (μ)**: Average value
  - $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Median**: Middle value when data is sorted
- **Mode**: Most frequently occurring value

### Measures of Variability
- **Variance (σ²)**: Average squared deviation from mean
  - $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2$
- **Standard Deviation (σ)**: Square root of variance
- **Range**: Difference between max and min values
- **Interquartile Range (IQR)**: Difference between 75th and 25th percentiles

### Applications in ML:
- **Data preprocessing**: Understanding data distribution before modeling
- **Feature scaling**: Normalizing features based on statistical properties
- **Outlier detection**: Identifying unusual data points

## Probability Theory

Probability quantifies uncertainty and randomness in data.

### Basic Concepts
- **Sample Space (Ω)**: Set of all possible outcomes
- **Event (A)**: Subset of sample space
- **Probability P(A)**: Number between 0 and 1 representing likelihood

### Probability Rules
1. **Addition Rule**: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
2. **Multiplication Rule**: P(A ∩ B) = P(A|B) × P(B)
3. **Complement Rule**: P(Aᶜ) = 1 - P(A)

### Conditional Probability
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Independence
Events A and B are independent if: P(A|B) = P(A)

### Applications in ML:
- **Probabilistic models**: Naive Bayes, Hidden Markov Models
- **Uncertainty quantification**: Bayesian neural networks
- **Decision making**: Expected value calculations

## Probability Distributions

Distributions describe how probability is spread across possible values.

### Discrete Distributions

#### Bernoulli Distribution
- **Use case**: Binary outcomes (success/failure)
- **Parameters**: p (probability of success)
- **PMF**: P(X = k) = p^k(1-p)^(1-k), k ∈ {0,1}

#### Binomial Distribution
- **Use case**: Number of successes in n independent trials
- **Parameters**: n (trials), p (success probability)
- **PMF**: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$

#### Poisson Distribution
- **Use case**: Number of events in fixed interval
- **Parameter**: λ (rate)
- **PMF**: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

### Continuous Distributions

#### Normal (Gaussian) Distribution
- **Most important distribution in ML**
- **Parameters**: μ (mean), σ² (variance)
- **PDF**: $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **Properties**: Bell-shaped, symmetric, 68-95-99.7 rule

#### Uniform Distribution
- **Use case**: Equal probability over an interval
- **Parameters**: a (minimum), b (maximum)
- **PDF**: $f(x) = \frac{1}{b-a}$ for x ∈ [a,b]

#### Exponential Distribution
- **Use case**: Time between events
- **Parameter**: λ (rate)
- **PDF**: $f(x) = \lambda e^{-\lambda x}$ for x ≥ 0

### Applications in ML:
- **Assumption validation**: Many algorithms assume normal distributions
- **Generative modeling**: Learning data distributions
- **Regularization**: Prior distributions in Bayesian methods

## Inferential Statistics

Inferential statistics help us make conclusions about populations from samples.

### Central Limit Theorem
**Key insight**: Sample means approach normal distribution as sample size increases

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

### Confidence Intervals
Range of values likely to contain true population parameter

95% CI for mean: $\bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$

### Standard Error
Standard deviation of sampling distribution: $SE = \frac{\sigma}{\sqrt{n}}$

### Applications in ML:
- **Model validation**: Confidence intervals for accuracy estimates
- **A/B testing**: Comparing model performance
- **Bootstrap sampling**: Estimating model uncertainty

## Bayes' Theorem

Fundamental theorem connecting conditional probabilities.

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

### Components
- **P(A|B)**: Posterior probability
- **P(B|A)**: Likelihood
- **P(A)**: Prior probability
- **P(B)**: Evidence

### Bayesian Thinking
- Start with prior beliefs
- Update beliefs with new evidence
- Result in posterior beliefs

### Applications in ML:
- **Naive Bayes classifier**: Direct application
- **Bayesian optimization**: Hyperparameter tuning
- **Bayesian neural networks**: Uncertainty in deep learning
- **Spam filtering**: Email classification

## Hypothesis Testing

Statistical method for making decisions about population parameters.

### Steps
1. **Formulate hypotheses**:
   - H₀: Null hypothesis (status quo)
   - H₁: Alternative hypothesis (what we want to prove)

2. **Choose significance level (α)**: Usually 0.05

3. **Calculate test statistic**

4. **Determine p-value**

5. **Make decision**: Reject H₀ if p-value < α

### Common Tests
- **t-test**: Comparing means
- **Chi-square test**: Testing independence
- **ANOVA**: Comparing multiple groups

### Type I and Type II Errors
- **Type I Error (α)**: Rejecting true null hypothesis
- **Type II Error (β)**: Failing to reject false null hypothesis
- **Power (1-β)**: Probability of correctly rejecting false null

### Applications in ML:
- **Feature selection**: Testing feature importance
- **Model comparison**: Statistical significance of performance differences
- **A/B testing**: Comparing algorithm variants

## Practice Problems

### Problem 1: Basic Probability
A dataset has 60% positive class and 40% negative class. If we randomly select 3 samples with replacement:
1. What's the probability all 3 are positive?
2. What's the probability exactly 2 are positive?

### Problem 2: Normal Distribution
Given a normally distributed variable with μ = 100 and σ = 15:
1. What percentage of values fall between 85 and 115?
2. What's the 95th percentile?

### Problem 3: Bayes' Theorem
A medical test is 95% accurate for detecting a disease that affects 1% of the population. If someone tests positive, what's the probability they actually have the disease?

### Problem 4: Hypothesis Testing
You're testing if a new ML model performs better than the baseline. The baseline has 80% accuracy. After testing your model on 100 samples, you get 85% accuracy. Is this statistically significant at α = 0.05?

### Problem 5: Central Limit Theorem
You have a population with μ = 50 and σ = 10. If you take samples of size 25:
1. What's the mean of the sampling distribution?
2. What's the standard error?
3. What's the probability a sample mean is greater than 52?

## Next Steps

After mastering these statistical concepts:
1. Practice with real datasets using Python libraries (scipy.stats, statsmodels)
2. Apply statistical thinking to ML problems
3. Study Bayesian machine learning methods
4. Explore statistical learning theory

## Resources for Further Learning

- **Books**:
  - "Think Stats" by Allen B. Downey
  - "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
  - "Bayesian Data Analysis" by Gelman et al.

- **Online Courses**:
  - Khan Academy Statistics
  - MIT 18.05 Introduction to Probability and Statistics
  - Coursera's "Introduction to Statistics"

- **Practice**:
  - Work through problems with Python/R
  - Analyze real datasets
  - Implement statistical tests from scratch

---

**Note**: Statistics is a vast field. This guide covers the fundamentals most relevant to machine learning. Depending on your specific interests (e.g., time series, experimental design), you may need to dive deeper into specialized topics.
