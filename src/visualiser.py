import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from . import risk_metrics

class Visualiser:
    
    def __init__(self, portfolio):
        self.portfolio = portfolio


    def plot_monte_carlo_cloud(self, results):
        plt.plot(results)
        plt.ylabel('Portfolio Value £')
        plt.xlabel('Days')
        plt.title('Monte Carlo Simulation of stock portfolio')
        plt.show()



    
    def plot_efficient_frontier(self, results):
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(
            results["Volatility"],
            results["Return"],
            c=results["Sharpe"],
            cmap="viridis",
            alpha=0.7
        )

        plt.colorbar(scatter, label="Sharpe Ratio")
        plt.xlabel("Volatility")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()



    def plot_weights_and_metrics(self, mc_best, mpt, labels):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        mc_weights = mc_best["weights"]

        wedges, _ = axes[0].pie(mc_weights, startangle=90)
        axes[0].set_title("Monte Carlo Best Sharpe Portfolio")

        mc_labels = [
            f"{lbl} - {w:.1%}" for lbl, w in zip(labels, mc_weights)
        ]
        axes[0].legend(
            wedges, mc_labels,
            loc="center left", bbox_to_anchor=(1, 0.5),
            fontsize=8
        )

        # Risk metrics
        mc_returns = self.portfolio.returns @ mc_weights
        mc_cum = (1 + mc_returns).cumprod()
        mc_var = risk_metrics.value_at_risk(mc_returns)
        mc_cvar = risk_metrics.conditional_var(mc_returns)
        mc_dd = risk_metrics.max_drawdown(mc_cum)
        mc_vol = risk_metrics.volatility(mc_returns)

        mc_text = (
            f"VaR(5%): {mc_var:.4f}\n"
            f"CVaR(5%): {mc_cvar:.4f}\n"
            f"Max DD: {mc_dd:.4f}\n"
            f"Volatility: {mc_vol:.4f}"
        )
        axes[0].text(0, -1.3, mc_text, ha="center", va="top", fontsize=10)


        avg_weights = np.ravel(mpt)

        wedges, _ = axes[1].pie(avg_weights, startangle=90)
        axes[1].set_title("MPT Avg Weights")

        # legend
        mpt_labels = [
            f"{lbl} - {w:.1%}" for lbl, w in zip(labels, avg_weights)
        ]
        axes[1].legend(
            wedges, mpt_labels,
            loc="center left", bbox_to_anchor=(1, 0.5),
            fontsize=8
        )

        mpt_returns = self.portfolio.returns @ avg_weights
        mpt_cum = (1 + mpt_returns).cumprod()
        mpt_var = risk_metrics.value_at_risk(mpt_returns)
        mpt_cvar = risk_metrics.conditional_var(mpt_returns)
        mpt_dd = risk_metrics.max_drawdown(mpt_cum)
        mpt_vol = risk_metrics.volatility(mpt_returns)

        mpt_text = (
            f"VaR(5%): {mpt_var:.4f}\n"
            f"CVaR(5%): {mpt_cvar:.4f}\n"
            f"Max DD: {mpt_dd:.4f}\n"
            f"Volatility: {mpt_vol:.4f}"
        )
        axes[1].text(0, -1.3, mpt_text, ha="center", va="top", fontsize=10)

        plt.tight_layout()
        plt.show()
