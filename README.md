# FIN 337: Team 12 Final Project Repo
## By: Kevin Chen, Ziggy Lee, Ryan Thomas, Jinyi Xu

This repository contains the full final project for FIN 377: Data Science for Finance.

In our project, we aim to look at the IPO market observing traditional IPOs and SPAC mergers. We analyze data from [Jay R. Ritter PhD](https://site.warrington.ufl.edu/ritter/) (University of Florida), [SPACInsider.com](https://www.spacinsider.com/), [wrds](https://wrds-www.wharton.upenn.edu/), stock market data, and SEC Filings(Form S-1 and Form 425). 

## Research Question

1. Broad Question:

- Our project aims to evaluate the effectiveness and efficiency of SPACs versus IPOs in accessing public capital markets, focusing on which method delivers better long-term value for companies and investors, especially given the recent surge in SPACs' popularity.

    Specific Research Question:
- Performance Comparison: How do the long-term financial performances of companies that go public via SPACs compare to those that conduct traditional IPOs?
- Cost Analysis: What are the total costs associated with going public through a SPAC versus an IPO?

2. Hypothesis

- Hypothesis: The total cost of going public via a SPAC is lower when measured immediately post-IPO but higher when considering long-term performance metrics.

3. Prediction Metrics

- Metrics of Success: The primary metric of success could be the long-term stock price performance. Secondary metrics might include market capitalization growth, and financial health metrics such as EBITDA growth, or debt-to-equity ratios.
Baseline for Comparison: Historical averages of post-IPO performance for both SPACs and IPOs as a baseline.

## Necessary Data

1.

2. Currently, we have Data from Jay R. Ritter PhD on traditional IPOs, information from SPACInsider.com regarding SPAC transactions, access to stock market data, SEC Filings (Forms S-1, 425) and historical data available through WRDS. What we need is more detailed post-IPO/SPAC financial performance data, updated and comprehensive data on SPAC sponsors and outcomes and market context data for the sample period.

3. 

4.

5. Steps on how we will transform the raw data into the final form: 

    **Cleaning and Preprocessing**: Check for and handle missing data, eliminate duplicate records, and unify formats from various sources.

    **Integration**: Combine different datasets using common identifiers such as company names or transaction dates.

    **Aggregation**: Compile data into annual or quarterly figures for trend analysis.
   