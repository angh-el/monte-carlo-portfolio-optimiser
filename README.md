# monte-carlo-portfolio-optimiser

# 📌 Introduction
This project is a Python toolkit for portfolio optimisation and risk analysis. It combines Modern Portfolio Theory with Monte Carlo simulations to derrive optimal portfolio allocations, compare them and visualise the results.

## Project goals:
- Explore optimal portfolio weights using Black-Litterman optimisation
- Run Monte Carlo Simulations to compare against MPT optimised portfolio weights
- Analyse portfolio performance using risk metrics (VaR, CVaR, max draw down and volatility)
- Provide visualisations for decision making

# ⚙️ How it works
The project is modular with each file handling a specific task:
- ``data_fetcher.py`` -> fetches historical asset prices (via yfinance)
- ``portfolio.py`` -> manages asset returns and stores protfolio data
- ``optimiser.py`` -> uses PyPortfolioOpt and the Black-Litterman model to calculate optimal weights
- ``simulator.py`` -> runs Monte Carlo simulations of random portfolios
- ``risk_metrics.py`` -> computes VaR, CVaR, max draw down and volatility
- ``visualiser.py`` -> creates plots to visualise the data

# 📈 Visualisations

<img width="795" height="675" alt="image" src="https://github.com/user-attachments/assets/97858179-5197-4ace-9344-5f98df9b4d83" />


<img width="985" height="826" alt="image" src="https://github.com/user-attachments/assets/66b4480d-5484-46d9-803d-610df8339c1e" />


<img width="1742" height="824" alt="image" src="https://github.com/user-attachments/assets/2cfe03e9-7168-4341-82a8-65179d5fc68f" />

