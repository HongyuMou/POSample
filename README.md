# POSample Package

## Introduction
The `POSample` package is designed for processing and analyzing data for the project of "Social_Mobility_and_Efficiency", particularly focusing on local approximations and extrapolations. The primary function, `POSample(csv_file_path)`, reads data from a CSV file, performs a series of analyses including OLS coefficients of the extrapolation model, figures indicating grid size, and plotting of conditional probabilities.

## Installation
To install the `POSample` package, first clone the repository from GitHub and then use pip to install it locally:

```bash 
git clone https://github.com/HongyuMou/POSample.git
cd POSample
git pull origin main # Update local repository to get the latest version
pip install .
```

## Usage
Here's a simple example of how to use the `POSample` package:

```python
from POSample import POSample

# Replace 'path_to_csv' with the path to your CSV file
csv_file_path = 'path_to_csv.csv'
POSample.POSample(csv_file_path)
```

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scipy
- statsmodels

Ensure these are installed in your environment to use `POSample`.

## Contributing
Contributions to the `POSample` package are welcome. Please submit pull requests to the GitHub repository.

## License
This project is licensed under the [MIT License](LICENSE).

## Example Usage



