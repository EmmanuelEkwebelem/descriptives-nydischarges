import pandas
import numpy
import researchpy 
import statsmodels
import seaborn
import scipy
import sapply

from tableone import TableOne, load_dataset
from pandas.plotting import scatter_matrix
from scipy import stats
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt

### Reading the data via API 
SPARCS= pandas.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')

### Removing Duplicate and Empty Cells 
SPARCS = SPARCS.drop_duplicates(keep='first')
SPARCS = SPARCS.dropna()

### Getting the shape, column names and data types of dataset
print(SPARCS.shape)
print(SPARCS.columns)
print(SPARCS.dtypes)

### Finding the "Summary Statistic" of each column
SPARCS.describe()
### ResearchPy Analysis 
researchpy.codebook(SPARCS)
print(researchpy.summarize(SPARCS))
print(researchpy.summary_cont(SPARCS[['age_group', 'gender', 'race', 'ethnicity', 'length_of_stay', 'type_of_admission', 'apr_severity_of_illness_code', 'apr_risk_of_mortality', 'total_charges', 'total_costs']]))
print(researchpy.summary_cat(SPARCS[['age_group', 'gender', 'race', 'ethnicity', 'length_of_stay', 'type_of_admission', 'apr_severity_of_illness_code', 'apr_risk_of_mortality', 'total_charges', 'total_costs']]))

### Comparing the correlation between columns in ascending order
print(SPARCS.corr().unstack().sort_values(ascending = False).drop_duplicates())

### Heatmap of correlation between columns
SPARCSCorr = SPARCS.corr()
filteredSPARCS = SPARCSCorr[((SPARCSCorr >= .5) | (SPARCSCorr <= -.5)) & (SPARCSCorr !=1.000)]
plt.figure(figsize=(30,10))
seaborn.heatmap(filteredSPARCS, annot=True, cmap="Greens")
plt.xticks(rotation=90)
plt.show()

### Scattermatrix to visualize data 
pandas.plotting.scatter_matrix(SPARCS)
plt.show()

#### TableOne Analysis
SPARCS_Columns = ['age_group', 'gender', 'race', 'length_of_stay', 'type_of_admission', 'apr_severity_of_illness_code', 'apr_risk_of_mortality']
SPARCS_Categorial = ['race','age_group', 'gender', 'type_of_admission', 'apr_risk_of_mortality']
SPARCS_NewLabels = {'apr_severity_of_illness_code' : 'Severity', 'apr_risk_of_mortality' : 'MortalityRisk', 'type_of_admission' : 'AdmissionType'}
SPARCS_Grouping = ['apr_severity_of_illness_code']

sparcs_df1_table1 = TableOne(SPARCS, columns=SPARCS_Columns, categorical=SPARCS_Categorial, groupby=SPARCS_Grouping, rename=SPARCS_NewLabels, pval=False)
print(sparcs_df1_table1.tabulate(tablefmt="fancy_grid"))