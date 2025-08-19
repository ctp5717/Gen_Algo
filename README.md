# AI Genetic Algorithm Trading Framework

### Overview

This project is a sophisticated and highly configurable Python-based framework for discovering, optimizing, and backtesting algorithmic trading strategies. It leverages a **Genetic Algorithm (GA)** to evolve strategy parameters, finding potentially profitable solutions over historical market data.

The core philosophy is a modular, "batteries-included" design that separates the strategy logic, data handling, and optimization engine. This allows for rapid experimentation and robust validation of new trading ideas.

### Key Features (Current Version 1.2)

  * **Genetic Algorithm Core:** Uses the `pygad` library to optimize strategy parameters across a multi-generational process.
  * **Dynamic Strategy Configuration:** All strategy rules, indicators, and parameters are defined in a single, easy-to-edit `config.py` file.
  * **Multi-Indicator Support:** The engine can build strategies using a confluence of indicators, including EMA, RSI, MACD, and Bollinger Bands, with on/off switches for each rule.
  * **Multi-Source Data Loader:** Fetches historical data from both `yfinance` and the **Binance.US** API, with automated caching.
  * **Automated Rolling Dates:** Intelligently calculates training and validation periods based on the current date and selected timeframe.
  * **Robust Backtesting:** Utilizes the `vectorbt` library for high-speed, vectorized backtesting.
  * **Advanced Risk Management:** Includes optimizable stop-loss, take-profit, trailing stop loss, and a static max hold period.
  * **Composite Fitness Function:** The GA optimizes for a blended score of multiple metrics (Sortino Ratio, Profit Factor, Max Drawdown) to find more balanced strategies.
  * **Multi-Asset Fitness with Robustness Penalty:** Evaluate strategies across a group of assets using the weighted mean minus λ·standard deviation of per-asset scores with a configurable trade floor.
  * **Automated Validation:** After optimization, the "champion" strategy is automatically tested on a separate, unseen out-of-sample dataset.
  * **Progress Tracking:** A live progress bar provides real-time feedback during optimization runs.

### Project Architecture

Of course. Here is a comprehensive breakdown of each file in our project and its specific role in the framework.

---
### `config.py` - The Control Panel
This is the single most important file for the user. It is the central "control panel" for the entire application, designed so that you can run vastly different experiments without ever touching the core logic files.

* **Key Responsibilities:**
    * **Data Source Selection:** Sets the `DATA_SOURCE` (`binance` or `yfinance`) and reads API credentials from environment variables.
    * **Asset & Timeframe:** Defines which asset to test (`SELECTED_ASSET_NAME`) and at what resolution (`TIMEFRAME`).
    * **Dynamic Date Calculation:** Intelligently calculates the rolling `TRAINING_PERIOD` and `VALIDATION_PERIOD` based on the current date and the selected timeframe, automatically respecting the data history limits of the chosen API.
    * **Strategy Definition:** Contains the `STRATEGY_RULES` dictionary, the heart of the system. This is where you build your trading strategy by combining indicator rules, setting their parameters, defining which parameters should be optimized as "genes," and using `is_active` flags to turn rules on or off.
    * **Risk Management:** Sets the `MAX_HOLD_PERIOD` for trades, now expressed
      as days converted into bars based on the selected `TIMEFRAME`.
    * **GA Tuning:** Holds all parameters for the Genetic Algorithm (`GA_POPULATION_SIZE`, `GA_NUM_GENERATIONS`, etc.).
    * **Fitness Criteria:** Defines the `FITNESS_WEIGHTS` for the composite score, telling the AI what characteristics of a "good" strategy to prioritize.
    * **Multi-Asset Settings:** The `MULTI_ASSET` block controls group evaluation including asset weights, dispersion penalty `lambda_dispersion`, trade-floor policies, zero-trade handling and per-asset trade requirements. A global `COVERAGE_THRESHOLD` setting decides how much historical data an asset must have to be included, and the `poor_score` value defines the sentinel fitness used when the trade floor fails.

---
### `data_loader.py` - The Data Handler
This module's sole responsibility is to fetch, clean, and cache market data.

* **Key Responsibilities:**
    * **Data Routing:** Acts as a "router" by checking the `DATA_SOURCE` in the config file and calling the appropriate private function to get data (e.g., `_get_binance_data`).
    * **API Connection:** Contains the specific logic for connecting to different APIs (Binance, yfinance).
    * **Data Standardization:** Cleans the data returned from different APIs into a single, standard format (a pandas DataFrame with a `DatetimeIndex` and capitalized column names: `Open`, `High`, `Low`, `Close`, `Volume`). This includes flattening complex `MultiIndex` columns.
    * **Caching:** Manages the `data_cache` directory. It saves a local copy of any downloaded data so that subsequent runs are nearly instantaneous, avoiding redundant API calls.

---
### `indicator_library.py` - The Toolbox
This file is a simple, clean, and expandable "toolbox" of functions for calculating technical indicators.

* **Key Responsibilities:**
    * Contains a separate, self-contained function for each indicator (e.g., `calculate_ema`, `calculate_rsi`, `calculate_macd`).
    * Uses the high-performance `pandas-ta` library to perform the actual mathematical calculations, ensuring speed and accuracy.
    * Provides a standardized interface where each function accepts price data and parameters, and returns the calculated indicator values.

---
### `strategy_engine.py` - The Logic Engine
This is the core processor that translates your ideas from the config file into actual trading signals.

* **Key Responsibilities:**
    * **Rule Interpretation:** It reads the `STRATEGY_RULES` dictionary from the config.
    * **Dynamic Indicator Calls:** It uses the `INDICATOR_MAPPING` dictionary to dynamically call the correct calculation functions from the `indicator_library.py` based on the active rules.
    * **Signal Generation:** It processes the `'condition'` logic for each rule (e.g., `'price_is_above_indicator'`, `'indicator_crosses_above_value'`) to generate a boolean Series of signals.
    * **Intelligent Column Selection:** For indicators that return multiple columns of data (like MACD or Bollinger Bands), it intelligently selects the correct column to use based on the condition type.
    * **Signal Combination:** It combines the signals from all active rules using the specified `combination_logic` (`AND`).
    * **Output:** It returns a final, single pandas Series of `True`/`False` entry signals to the backtester.

---
### `fitness.py` - The GA's Judge
This module is the critical bridge between the Genetic Algorithm and our trading logic. It's responsible for evaluating how "fit" any given strategy is.

* **Key Responsibilities:**
    * **Gene Injection:** It takes a "solution" (a list of parameter values) from the GA and uses the `gene_map` to precisely inject those values into a temporary copy of the `STRATEGY_RULES`.
    * **Backtest Execution:** It calls the `strategy_engine` to get entry signals for the injected strategy, then runs a high-speed backtest using `vectorbt`, applying all exit logic (stop loss, trailing stop, take profit, max hold time).
    * **Performance Scoring:** It calculates the **composite fitness score** based on the backtest results and the `FITNESS_WEIGHTS` defined in the config file.
    * **Output:** It returns a single number (the fitness score) to the Genetic Algorithm, which the GA uses to rank that solution and guide the evolution process.

---
### `main.py` - The Orchestrator
This is the main entry point of the application. It controls the entire end-to-end workflow from setup to final analysis.

* **Key Responsibilities:**
    * **Initialization:** Imports all other modules and reads all settings from `config.py`.
    * **Gene Parsing:** Uses the `parse_genes_from_config` utility from `gene_parser.py` to scan the `STRATEGY_RULES`, find all active genes, and build the necessary `gene_space` and `gene_map` data structures required by `PyGAD`.
    * **Data Loading:** Calls the `data_loader` to fetch the **training data**.
    * **GA Execution:** Configures and launches the `pygad.GA` instance, enabling parallel processing and hooking in our custom `fitness` function and `on_generation` progress bar.
    * **Results Display:** Prints a summary of the best solution found by the GA.
    * **Handoff to Analysis:** Automatically calls the `analysis.py` module to perform the final validation run on the "champion" strategy.

---
### `analysis.py` - The Reporter
This module's purpose is to provide a final, unbiased report on the performance of the single best strategy discovered by the GA.

* **Key Responsibilities:**
    * **Out-of-Sample Testing:** It loads a completely separate, **unseen validation dataset** based on the `VALIDATION_PERIOD` in `config.py`.
    * **Champion Backtest:** It re-runs the backtest **one time** using the winning set of parameters found by the GA.
    * **Statistical Reporting:** It uses `vectorbt` to generate and print a detailed table of performance statistics (Total Return, Max Drawdown, Win Rate, etc.).
    * **Visualization:** It generates and displays a plot of the strategy's equity curve against the benchmark "buy and hold" return for the validation period.

### Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a file named `requirements.txt` with the content below and run the installation command.

    **`requirements.txt` file:**

    ```text
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
    ```

    **Installation Command:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Add API Keys:**

      * Set the following environment variables before running the framework:

        - `BINANCE_API_KEY`
        - `BINANCE_API_SECRET`
        - `BINANCE_TLD` (optional, defaults to `us`)

      The `config.py` file automatically reads these values so no manual editing
      of the source code is required.

### How to Use

1.  **Configure Your Experiment:** Open `config.py`.

      * Set the `SELECTED_ASSET_NAME`, `TIMEFRAME`, and `DATA_SOURCE`.
      * Go to the `STRATEGY_RULES` dictionary. Use the `'is_active': True/False` flags to choose which indicator conditions to include in your strategy.
      * Adjust the `low` and `high` ranges for any genes you want to optimize.

2.  **Run the Optimizer:** Execute the `main.py` script from your terminal.

    ```bash
    python3 main.py
    ```

3.  **Analyze the Results:** The script will first run the optimization, printing a live progress bar. At the end, it will display a plot of the GA's learning curve, print the optimal parameters, and then automatically run the final analysis on the unseen validation data, printing a full statistical report and displaying the final equity curve.

When multi-asset analysis runs, an overview chart is also written to
`reports/{run_ts}/overview_{sha}.png`, where `run_ts` is the timestamp of the
run and `sha` is the short commit hash. Set `DISABLE_PNG_REPORTS=1` to skip
writing this file.

### Project Roadmap

Of course. Here is the complete project roadmap, with thorough descriptions for all the features we have not yet implemented.

### **Project Roadmap**

#### **V1.0: Minimum Viable Product (MVP) - ✅ Complete**
* A stable, single-asset optimization framework with core modular architecture and multi-core processing.

---
#### **V1.1: Quality of Life & Core Refinements - ✅ Complete**
* **Progress Tracking:** Real-time console updates during GA optimization.
* **Centralized Timeframe Configuration:** A single `TIMEFRAME` setting in `config.py` controls all modules.
* **Automated Rolling Dates:** The config file intelligently calculates training/validation periods, respecting API data limits.
  * **Maximum Trade Hold Duration:** A key risk parameter (`MAX_HOLD_PERIOD`) is
    now calculated as days converted into bars, ensuring consistency across
    different intraday timeframes.
* **Composite Fitness Function:** The AI optimizes for a blended score of multiple performance metrics (Sortino, Profit Factor, etc.).

---
#### **V1.2: Strategy Enhancement & Robustness**
* **Status:** **In Progress**
* **Features:**
    * **Expanded Indicator Library:** ✅ Complete.
    * **Advanced Exit Logic (Trailing Stops):** ✅ Complete.
    * **Walk-Forward Validation:** ✅ Complete.
    * **Tune GA Hyperparameters:** ✅ Complete.

---
#### **V2.0: Major Architectural Upgrade**
* **Status:** **Planned**
* **Features:**
    * **Portfolio-Level Optimization Engine:** (Planned)
        * *This is a major upgrade to the core engine. Instead of optimizing a strategy on a single asset, it will allow the GA to run backtests on a **basket of multiple assets** (e.g., BTC, ETH, and SOL) simultaneously. This is a powerful defense against overfitting, as it forces the AI to find a single, robust set of rules that performs well across different market behaviors, rather than a fragile strategy that only works on one asset's specific history.*
    * **Advanced Combination Logic:** (Planned)
        * *This is an upgrade to the `strategy_engine.py` to handle more complex ways of combining indicator signals. Currently, our engine can only combine entry rules with a logical `AND`. This feature will add support for **`OR`** logic (e.g., "enter if EMA is bullish OR RSI is oversold") and **`VOTE`** systems (e.g., "enter if at least 3 out of 5 conditions are true"), dramatically increasing the strategic flexibility of the framework.*
    * **Complete Indicator Library:** (Planned)
        * *This is the process of building out the remaining functions in `indicator_library.py` to include all 25 indicators we originally planned. This gives the AI the widest possible set of tools to build strategies with, increasing the potential for discovering novel patterns.*

---
#### **V3.0: Advanced Framework & Future Vision**
* **Status:** **Planned**
* **Features:**
    * **Genetic Programming (GP):** (Planned)
        * *This is a significant evolution beyond our current GA. Instead of just optimizing the **parameters** of a fixed strategy, GP would allow the AI to **build the strategy structure itself**. The "genes" would become the indicators and logical operators (e.g., `EMA`, `RSI`, `>`, `AND`). The AI would evolve entire trading rule trees from scratch, allowing it to discover completely novel strategies that a human might never design.*
    * **Graphical User Interface (GUI):** (Planned)
        * *This involves building a user-friendly, visual interface for the framework using a library like Streamlit. It would allow you to change `config.py` settings with buttons and dropdowns, launch optimization runs, and see the results and plots displayed in an interactive dashboard without having to directly edit code.*
    * **Live Trading Integration:** (Planned)
        * *This is the final step to connect the framework to the real world. We would build a module that connects to a broker's API (like Binance.US) and can automatically place paper or live trades based on the signals generated by a "champion" strategy.*

### License

This project is released under the [MIT License](LICENSE).

### Disclaimer

This framework is an educational tool for research and quantitative analysis. It is not financial advice. All trading involves significant risk, and any strategies developed with this tool should be thoroughly tested and understood before any capital is risked. Past performance is not indicative of future results.
