import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.stats as stats

# === Passo 1: Coleta de Dados (corrigido) ===
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Baixar apenas preços de fechamento ajustados para simplificar
data = yf.download(tickers, start='2020-01-01', end='2023-01-01', auto_adjust=True)['Close']

# Remover linhas com dados faltantes
data = data.dropna()

# Listar tickers como strings simples
tickers = data.columns.tolist()

# === Passo 2: Cálculo de Retorno e Risco ===
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252  # Retorno anualizado
cov_matrix = returns.cov() * 252  # Covariância anualizada

# === Passo 3: Modelo de Markowitz ===
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def constraint(weights):
    return np.sum(weights) - 1

num_assets = len(mean_returns)
initial_weights = np.ones(num_assets) / num_assets
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = {'type': 'eq', 'fun': constraint}

result = minimize(portfolio_volatility, initial_weights, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x

# === Passo 4: Cálculo do Value at Risk (VaR) ===
portfolio_return = np.dot(optimal_weights, mean_returns)
portfolio_volatility_value = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

confidence_level = 0.95
z_score = stats.norm.ppf(confidence_level)
var = portfolio_return - z_score * portfolio_volatility_value

# === Passo 5: Visualizações ===
plt.bar(tickers, optimal_weights, color='skyblue')
plt.xlabel('Ativos')
plt.ylabel('Pesos Otimizados')
plt.title('Alocação Ótima de Ativos')
plt.show()

# Fronteira Eficiente
risks = []
returns_list = []
for _ in range(1000):
    weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
    risks.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    returns_list.append(np.dot(weights, mean_returns))

plt.scatter(risks, returns_list, c=np.array(returns_list) / np.array(risks), cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risco (Volatilidade)')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente')
plt.scatter(portfolio_volatility_value, portfolio_return, c='red', marker='*', s=200, label='Portfólio Otimizado')
plt.legend()
plt.show()

# === Resultados ===
print("Pesos ótimos:", optimal_weights)
print(f"Retorno esperado do portfólio: {portfolio_return:.2%}")
print(f"Volatilidade do portfólio: {portfolio_volatility_value:.2%}")
print(f"Value at Risk (95%): {var:.2%}")