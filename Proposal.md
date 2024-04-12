# FIN 337: Team 12 Final Project Repo
## By: Kevin Chen, Ziggy Lee, Ryan Thomas, Jinyi Xu

This repository contains the full final project for FIN 377: Data Science for Finance.

In our project, we aim to look at the IPO market observing traditional IPOs and SPAC mergers. We analyze data from [Jay R. Ritter PhD](https://site.warrington.ufl.edu/ritter/) (University of Florida), [SPACInsider.com](https://www.spacinsider.com/), [wrds](https://wrds-www.wharton.upenn.edu/), stock market data, and SEC Filings(Form S-1 and Form 425). 

# Research Question

Research Question:

- How can we reliably determine the most suitable capital market approach between SPACs and IPOS for companies and investors, considering their long-term value implications? Particularly amidst the current surge in SPACs popularity, what indicators can accurately distinguish firms that are best suited for either a SPAC or IPO. 
 

Specific Research Question:
- Performance Comparison: How do the long-term financial performances of companies that go public via SPACs compare to those that conduct traditional IPOs?
- Cost Analysis: What are the total costs associated with going public through a SPAC versus an IPO?
- Predictive Modeling: Given a specific input of firm features, we can use KNN nearest neighbors to identify a similar firm that used either a SPAC or IPO

Measuremnt Variables: 
- Market Capitalization
- Growth Metrics: Revenue, Earnings, ROA...etc
- Industry Sector
- Investment Sentiment

## Hypothesis

- *Using KNN nearest neighbor analysis, we anticipate that the firm characteristics described in our measurment variables will significantly contribute to accurately predicting whether a firm is better suited for a SPAC or an IPO. Additionally, we hope to evelaute the total cost of going public via SPAC is lower when measured immediately post-IPO but higher when considering long-term performance metrics.*

## Prediction Metrics

Initial Statement:
*Within our results we hope to maximize specific metrics such as accuracy, precision and recall, and the F1 Score.*

Metrics of Success:
1. Accuracy: the proportion of correctly classified firms out of the total number of firms in the dataset. Maximizing this value will indicate the effectiveness of our model in making correct predictions.
2. 


The primary metric of success could be the long-term stock price performance. Secondary metrics might include market capitalization growth, and financial health metrics such as EBITDA growth, or debt-to-equity ratios.
Baseline for Comparison: Historical averages of post-IPO performance for both SPACs and IPOs as a baseline.

# Necessary Data

1. 

2. Currently, we have Data from Jay R. Ritter PhD on traditional IPOs, information from SPACInsider.com regarding SPAC transactions, access to stock market data, SEC Filings (Forms S-1, 425) and historical data available through WRDS. What we need is more detailed post-IPO/SPAC financial performance data, updated and comprehensive data on SPAC sponsors and outcomes and market context data for the sample period.

3. 

4. 

5. Steps on how we will transform the raw data into the final form: 

    **Cleaning and Preprocessing**: Check for and handle missing data, eliminate duplicate records, and unify formats from various sources.

    **Integration**: Combine different datasets using common identifiers such as company names or transaction dates.

    **Aggregation**: Compile data into annual or quarterly figures for trend analysis.

# Dashboard Setup

1. 