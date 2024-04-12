# FIN 337: Team 12 Final Project Repo
## By: Kevin Chen, Ziggy Lee, Ryan Thomas, Jinyi Xu

This repository contains the full final project for FIN 377: Data Science for Finance.

In our project, we aim to look at the IPO market observing traditional IPOs and SPAC mergers. We analyze data from [Jay R. Ritter PhD](https://site.warrington.ufl.edu/ritter/) (University of Florida), [SPACInsider.com](https://www.spacinsider.com/), [wrds](https://wrds-www.wharton.upenn.edu/), stock market data, and SEC Filings(Form S-1 and Form 425). 

# Research Proposal: 
# Predictive Modeling for Optimal Capital Market Selection: A Comparative Analysis of SPACs and IPOs

Research Question:

- How can we reliably determine the most suitable capital market approach between SPACs and IPOS for companies and investors, considering their long-term value implications? Particularly amidst the current surge in SPACs popularity, what indicators can accurately distinguish firms that are best suited for either a SPAC or IPO. 
 

Specific Research Question:
- Performance Comparison: How do the long-term financial performances of companies that go public via SPACs compare to those that conduct traditional IPOs?
- Cost Analysis: What are the total costs associated with going public through a SPAC versus an IPO?
- Predictive Modeling: Given a specific input of firm features, we can use KNN nearest neighbors to correctly classify a firm as best suited for a SPAC or IPO

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
2. Precision: measures the proportion of correctly classified SPAC or IPO firms out of all firms classified as SPAC or IPO. 
3. Recall: measures the proportion of correctly classfied SPAC or IPO firms out of all actual SPAC or IPO firms. 
4. F1 Score: This is the harmonic mean of recall and precision. A higher F1 score will indicate a better balance between the two measurments. 

Baseline for Comparison: We hope to aim for an accuracy of 70% or higher. This would demonstrate the added value of our predictive modeling approach. We also want our data to reflect historical averages of post-IPO performance for both SPACs and IPOs as a baseline.

# Necessary Data

1. The final dataset needs to contain all IPOs (traditional and SPAC) in at least the past 10-15 years--but ideally all available data. This would include firm characteristics (sector, relevent financial ratios, year, traditional vs SPAC IPO, etc.)
    - An observation in our dataset will be an IPO
    - The sample period will be all IPOs
    - We anticipate some restructions in the sample surrounding availability of data. An example of this would be certain financial ratios not being recorded before a given year.
    - Variables that are necessary for this exploration:
        - Firm
        - Year
        - Sector
        - Traditional IPO vs SPAC IPO
        - Financial ratios across every observation to compare firms
        - Cost structre analysis
        - Stock returns
    - Variables we would like to have:
        - Success rate (measuered by performance after IPO)
        - Sector growth indicators (is a certain company taking advantage of favorable SPAC offerings?)

2. Currently, we have Data from Jay R. Ritter PhD on traditional IPOs, information from SPACInsider.com regarding SPAC transactions, access to stock market data, SEC Filings (Forms S-1, 425) and historical data available through WRDS. What we need is more detailed post-IPO/SPAC financial performance data, updated and comprehensive data on SPAC sponsors and outcomes and market context data for the sample period.

3. 

4. 

5. Steps on how we will transform the raw data into the final form: 

    **Cleaning and Preprocessing**: Check for and handle missing data, eliminate duplicate records, and unify formats from various sources.

    **Integration**: Combine different datasets using common identifiers such as company names or transaction dates.

    **Aggregation**: Compile data into annual or quarterly figures for trend analysis.

# Dashboard Setup
## Structure: 
- Our initial report will be displayed along with our findings and conclusive results. Additionally, users will find input fields adjacent to a scatter plot that visualizes predictions. This dashboard will effectively produce predictied results related to the user inputs and accurately classify whether a firm is best suited for an IPO or SPAC. 
## User Inputs: 
- Year
- Industry Sector
- Market Capitalization
- Profitability Metrics: ROA, ROE...etc
- Funding Needs
- Revenue Growth Rate
## Output: 
- Our dashboard will intake user inputs and utilize KNN nearest neighbors modeling to accurately classify whether a firm is best suited for a SPAC or IPO. Furthermore, we will visually represent this classification by plotting the firm's nearest neighbors on a scatter plot.  If the inputs strongly resemble characteristics associated with an IPO rather than a SPAC, the firm will be positioned close to its related entities. 