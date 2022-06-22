# National attribution of historical climate damages

This repository provides replication data and code for the paper "National attribution of historical climate damages," by Christopher W. Callahan and Justin S. Mankin, published in _Climatic Change_ (https://doi.org/10.1007/s10584-022-03387-y).

### Overview

The repository is organized into **Scripts/**, **Figures/**, and **Data/** folders.

- **Scripts/**: All code required to reproduce the findings of our work is included in this folder. Most of the code is provided in Jupyter notebooks, except for major scripts like *Calculate\_Damages.py*, which requires batch processing on a high-performance computing cluster.

- **Figures/**: The final figures, both in the main text and the supplement, are included in this folder.

- **Data/**: This folder includes intermediate and processed summary data that enable replication of all the figures and numbers cited in the text. The full data associated with the project amount to 7TB in total, so we do not provide this data in full. Should you desire any of this underlying data, feel free to contact me at _Christopher.W.Callahan.GR (at) dartmouth (dot) edu_ and I will happily work to organize a mass data transfer.

### Details of specific scripts

- *Construct_Panel.ipynb*
- *FaIR_Attribution_Ensemble.ipynb*
- *FaIR_GHGEmissions_Input.ipynb*
- *Country_Pattern_Scaling.ipynb*: This script reads global mean and country-level temperatures from the two climate model ensembles (CMIP6 and CESM2-LE) and calculates pattern scaling coefficients for each country in each model/realization. The output is an array of coefficients that describe the sensitivity of each country's temperature to global forcing, with uncertainty from both model structure (CMIP6) and internal variability (CESM2-LE).
- *Calculate_Damages.py*: This is the workhorse script of the paper. This script loads the observed economic and climate data, country-specific pattern scaling coefficients, FaIR output, and damage function parameters, and uses them to calculate the economic damages in each country attributable to each other country. The script is written to take command-line input for the specific accounting scheme and time period required. So if you wanted to calculate attributable damages over 1990-2014 using consumption-based emissions and the default short-run damage function, you'd type *python Calculate_Damages.py consumption BHMSR 1990 2014 1990 2014*. The repetitive years (typing "1990 2014" twice) are to enable you to calculate economic damages and country-level emissions contributions from FaIR over two different time periods (e.g., economic damages over 1990-2014 based on U.S. emissions contributions from 1960-2014), but in practice these are always the same pair of years.
- *Process_Damages.py*
- *Fig1.ipynb*, *Fig2.ipynb*, *Fig3.ipynb*, *Fig4.ipynb*: These scripts load processed data and create the plots that you can find the paper. No post-processing of figures required!

Christopher Callahan

June 2022
