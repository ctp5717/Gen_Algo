AI Genetic Algorithm Trading Framework
Overview

This project is a sophisticated and highly configurable Python-based framework for discovering, optimizing, and backtesting algorithmic trading strategies. It leverages a Genetic Algorithm (GA) to evolve strategy parameters, finding potentially profitable solutions over historical market data.

The core philosophy is a modular, "batteries-included" design that separates the strategy logic, data handling, and optimization engine. This allows for rapid experimentation and robust validation of new trading ideas.

Key Features (Current Version 1.2)

Genetic Algorithm Core: Uses the pygad library to optimize strategy parameters across a multi-generational process.

Dynamic Strategy Configuration: All strategy rules, indicators, and parameters are defined in a single, easy-to-edit config.py file.

Multi-Indicator Support: The engine can build strategies using a confluence of indicators, including EMA, RSI, MACD, and Bollinger Bands.

Flexible Rule Engine: Easily enable or disable individual trading rules using 'is_active' flags in the configuration.

Multi-Source Data Loader: Fetches historical data from both yfinance and the Binance.US API, with automated caching to speed up subsequent runs.

Automated Rolling Dates: Intelligently calculates training and validation periods based on the current date and selected timeframe, respecting API data limits.

Robust Backtesting: Utilizes the vectorbt library for high-speed, vectorized backtesting.

Advanced Risk Management: Includes optimizable stop-loss, take-profit, trailing stop loss, and a static max hold period for trades.

Composite Fitness Function: The GA optimizes for a blended score of multiple metrics (Sortino Ratio, Profit Factor, Max Drawdown) to find more balanced and robust strategies.

Automated Validation: After optimization, the "champion" strategy is automatically tested on a separate, unseen out-of-sample dataset to check for overfitting.

Project Architecture

The framework is broken down into several distinct modules, each with a specific responsibility:

main.py: The main orchestrator that runs the entire optimization and analysis pipeline.

config.py: The central control panel. This is the primary file you will edit to define your strategies and experiments.

data_loader.py: Handles all data fetching and caching from external APIs (Binance, yfinance).

indicator_library.py: The "toolbox" containing the calculation logic for all technical indicators.

strategy_engine.py: The core logic engine that interprets the rules from config.py and generates trading signals.

fitness.py: The bridge between the GA and the backtester. It evaluates the performance ("fitness") of each strategy.

analysis.py: Performs a deep-dive analysis on the final "champion" strategy and generates performance reports.

Setup and Installation

Clone the Repository:

Bash
git clone <your-repo-url>
cd <your-repo-name>
Create a Virtual Environment (Recommended):

Bash
python3 -m venv venv
source venv/bin/activate
Install Dependencies: Create a file named requirements.txt with the content below and run the installation command.

requirements.txt file:

Plaintext
pygad
vectorbt
pandas
numpy
yfinance
python-binance
pandas-ta
setuptools
python-dateutil
matplotlib
Installation Command:

Bash
pip install -r requirements.txt
Add API Keys:

Open the config.py file.

Find the API_KEYS dictionary.

Enter your Binance.US API key and secret.

How to Use

Configure Your Experiment: Open config.py.

Set the SELECTED_ASSET_NAME, TIMEFRAME, and DATA_SOURCE.

Go to the STRATEGY_RULES dictionary. Use the 'is_active': True/False flags to choose which indicator conditions to include in your strategy.

Adjust the low and high ranges for any genes you want to optimize.

Set your desired GA_... parameters for a "quick test" or a "discovery run."

Run the Optimizer: Execute the main.py script from your terminal.

Bash
python3 main.py
Analyze the Results: The script will first run the optimization, printing a live progress bar. At the end, it will display a plot of the GA's learning curve, print the optimal parameters it found, and then automatically run the final analysis on the unseen validation data, printing a full statistical report and displaying the final equity curve.

Project Roadmap

V1.2 (Current): Focus on Strategy Enhancement (Walk-Forward Validation, GA Tuning).

V2.0 (Planned): Major architectural upgrades, including a Portfolio-Level Optimization Engine and more advanced rule combination logic (OR, VOTE).

V3.0 (Future): Long-term goals including Genetic Programming (AI-driven strategy creation), a GUI, and live trading integration.

Disclaimer

This framework is an educational tool for research and quantitative analysis. It is not financial advice. All trading involves significant risk, and any strategies developed with this tool should be thoroughly tested and understood before any capital is risked. Past performance is not indicative of future results.
