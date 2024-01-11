# `POSample` Package in Python

## Introduction
The `POSample` package is designed for processing and analyzing data for the project of "Social_Mobility_and_Efficiency", particularly focusing on local approximations and extrapolations. The primary function, `POSample(csv_file_path)`, reads data from a CSV file, performs a series of analyses for each program, including OLS regressions of the extrapolation model, showing grid size, and plotting of conditional probabilities.

------

**Date:** Jan 7, 2024

**Version:** 0.3 ([Github](https://github.com/HongyuMou/POSample))

**Authors:** Benjamin U. Friedrich, Martin B. Hackmann, Hongyu Mou

## Installation
To install the `POSample` package, first open the terminal, clone the repository from GitHub, and then use pip to install it locally:

```bash 
git clone https://github.com/HongyuMou/POSample.git
cd POSample
pip install .
```

If the package was updated recently, you might need to reinstall it to get the latest version. Use:

```bash 
pip install --upgrade POSample
```

## Usage

Here's a simple example of how to use the `POSample` package, which works in Jupyter environment:

```python
from POSample import POSample

# Replace 'path_to_csv' with the path to your CSV file
csv_file_path = 'path_to_csv.csv'
POSample.POSample(csv_file_path)
```

## Dependencies
- numpy: version 1.22.4
- pandas: version 1.3.4
- seaborn: version 0.11.2
- matplotlib: version 3.4.3
- scipy: version 1.10.1
- scikit-learn: version 0.24.2
- statsmodels: version 0.13.5

Ensure these are installed in your environment to use `POSample`.

## Required Variables

The variables in the csv file can be:

| Variable Name            | Description                                                  |
| :----------------------- | ------------------------------------------------------------ |
| `applicant_id`           | Unique identifier for each applicant across the entire dataset. In the simulated example file *POsampleDGP_simul_example.csv*, I use a combination of the program index `k` and a unique identifier within each program. |
| `program_id`             | Program ID to identify the program it represents.            |
| `Available`              | Dummy variable to isolate appliers who are in the potential pool for each program. |
| `Applied_Q2`             | Dummy varibale where 1 indicates Quota 2 appliers.           |
| `Admitted_Q2`            | Dummy varibale with 1 indicating that the applier is admitted by this program through Quota 2. In the simulated example file *POsampleDGP_simul_example.csv*, `Admitted_Q2 = 1` if `S > 0 & Applied_Q2 == 1`. |
| `Admitted_Q1`            | Dummy varibale with 1 indicating that the applier is admitted by this program through Quota 1. In the simulated example file *POsampleDGP_simul_example.csv*, `Admitted_Q1 = 1` if `gpa >= GPA_cutoff`. |
| `GPA_cutoff`             | GPA cutoff (in GPA level) for admission to this program through Quota 1. It varies across different programs. |
| `gpa`                    | GPA level for Quota 1 appliers.                              |
| `percentile_GPA_applyQ1` | GPA percentile for Quota 1 appliers within each program, ranging from 0 to 1. |
| `background`             | Dummy variable for family background. For example, 1 indicates "parent with college degree". |
| `S`                      | Score level for Quota 2 appliers. It would be missing if the applicant does not apply to Quota 2. |
| `percentile_S_applyQ2`   | Score percentile for Quota 2 appliers within each program, ranging from 0 to 1. |
| `Y`                      | Outcome variable such as the 10 years income.                |
| `Q2` (optional)          | It is used to define `Applied_Q2` in the simulated example file *POsampleDGP_simul_example.csv*: `Applied_Q2 = 1` if `Q2 > 0`. |
| `Applied_Q1` (optional)  | Dummy varibale where 1 indicates Quota 1 appliers, which is also the universe of the whole sample in both the real dataset and the simulated data. Note we have `Applied_Q1 = 1` for each observation. |



## Contributing

Contributions to the `POSample` package are welcome. Please submit pull requests to the GitHub repository or report bugs to hongyumou@g.ucla.edu.



