# National attribution of historical climate damages

This repository provides replication data and code for the paper "National attribution of historical climate damages," by Christopher W. Callahan and Justin S. Mankin, published in _Climatic Change_ (https://doi.org/10.1007/s10584-022-03387-y).

We've built a website to provide much of the data that arose from this project: https://rcweb.dartmouth.edu/CMIG/national_attribution_2022/prod/. You can download the effects of one emitting country on each other country, or the effects of every other country on a single chosen country, by following the instructions on the landing page. Don't hesitate to contact me with questions if issues arise.

### Overview

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Scripts/**: All code required to reproduce the findings of our work is included in this folder. Most of the code is provided in Jupyter notebooks, except for major scripts like *Calculate\_Damages.py*, which requires batch processing on a high-performance computing cluster.

- **Figures/**: The final figures, both in the main text and the supplement, are included in this folder.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of all the figures and numbers cited in the text. The full data associated with the project amount to 7TB in total, so we do not provide this data in full. Should you desire any of this underlying data, feel free to contact me at _Christopher.W.Callahan.GR (at) dartmouth (dot) edu_ and I will be happy to organize a mass data transfer.

### Details of specific scripts

- *Construct_Panel.ipynb*: This script puts together the panel dataset of country-average temperature, precipitation, and economic output for use in calculating the empirical temperature-growth parameters.
- *Temp_Growth_Regression.R*: This script uses the empirical climate and economic data to calculate parameters relating fluctuations in temperature to fluctuations in economic growth, using multiple functional forms and dataset choices. The main approach follows Burke et al. (2015). [See here: https://www.nature.com/articles/nature15725]
- *FaIR_GHGEmissions_Input.ipynb*: This script processes and constructs the country-level emissions datasets for input into the simple climate model FaIR.
- *FaIR_Attribution_Ensemble.ipynb*: This script uses the Finite amplitude Impulse-Response (FaIR) model to simulate the global mean temperature response to total historical emissions as well as emissions when individual countries are subtracted.
- *Country_Pattern_Scaling.ipynb*: This script reads global mean and country-level temperatures from the two climate model ensembles (CMIP6 and CESM1-LE) and calculates pattern scaling coefficients for each country in each model/realization. The output is an array of coefficients that describe the sensitivity of each country's temperature to global forcing, with uncertainty from both model structure (CMIP6) and internal variability (CESM1-LE).
- *Calculate_Damages.py*: This is the workhorse script of the paper. This script loads the observed economic and climate data, country-specific pattern scaling coefficients, FaIR simulation output, and damage function parameters, and uses them to calculate the economic damages in each country attributable to each other country. The script is written to take command-line input for the specific accounting scheme and time period required. So if you wanted to calculate attributable damages over 1990-2014 using consumption-based emissions and the default short-run damage function, you'd type *python Calculate_Damages.py consumption BHMSR 1990 2014 1990 2014*. The repetitive years (typing "1990 2014" twice) are to enable you to calculate economic damages and country-level emissions contributions from FaIR over two different time periods (e.g., economic damages over 1990-2014 based on U.S. emissions contributions from 1960-2014), but in practice these are always the same pair of years. Importantly, this script is extremely computationally intensive. In production, it was run on a high-performance computing cluster over the course of several weeks. Running the attribution calculation for a single accounting method and time period will likely take >10 days, since there are in excess of 1.5 trillion data points involved.
- *Process_Damages.py*: This script takes the raw output from _Calculate\_Damages.py_, calculates summary statistics, and writes it out into a more manageable form for use in plotting. _This script will not work because the raw data are not provided due to large file sizes. If you would like these data, please contact the authors._
- *Process_Damages_Uncertainty.py*: This script takes the raw output from _Calculate\_Damages.py_, partitions the various sources of uncertainty, and writes that uncertainty partitioning data out into a more manageable form for use in plotting the supplementary figures. _This script will not work because the raw data are not provided due to large file sizes. If you would like these data, please contact the authors._
- *Process_Damages_byPercentile.py*: This script takes the raw output from _Calculate\_Damages.py_, averages damages across the income quantiles, and writes that data out into a more manageable form for use in plotting Figure 4. _This script will not work because the raw data are not provided due to large file sizes. If you would like these data, please contact the authors._
- *Fig1.ipynb*, *Fig2.ipynb*, *Fig3.ipynb*, *Fig4.ipynb*, *Supplemental_Figs.ipynb*: These scripts load processed data and create the plots that you can find the paper. No post-processing of main text figures required!

Christopher Callahan

July 2022
