# <p align="center">K-Means-Clustering using RFM variables to Segment Customers

For this project, we will be exploring K means algorithm to segment customers based on RFM model. The dataset comes from UCI machine learning repository - Online Retail II dataset.

## Data Loading and Initial Cleaning and Exploration
```bash
$ Loading, Initial Exploration and Cleaning...Loading Excel File...
Sheet Names: ['Year 2009-2010', 'Year 2010-2011']
Number of Sheets: 2
Cleaning Data...
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1067371 entries, 0 to 1067370
Data columns (total 8 columns):
 #   Column       Non-Null Count    Dtype
---  ------       --------------    -----
 0   Invoice      1067371 non-null  object
 1   StockCode    1067371 non-null  object
 2   Description  1062989 non-null  object
 3   Quantity     1067371 non-null  int64
 4   InvoiceDate  1067371 non-null  datetime64[ns]
 5   Price        1067371 non-null  float64
 6   Customer ID  824364 non-null   float64
 7   Country      1067371 non-null  object
dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
memory usage: 65.1+ MB
None
Count of Null Rows before Removal:  243007
Count of Nulls in Columns after Removal:  Invoice        0
StockCode      0
Description    0
Quantity       0
InvoiceDate    0
Price          0
Customer ID    0
Country        0
dtype: int64
Number of Records after Null Removal:  824364
Number of records before removing duplicates:  Invoice        824364
StockCode      824364
Description    824364
Quantity       824364
InvoiceDate    824364
Price          824364
Customer ID    824364
Country        824364
dtype: int64
Number of records after removing duplicates:  Invoice        797885
StockCode      797885
Description    797885
Quantity       797885
InvoiceDate    797885
Price          797885
Customer ID    797885
Country        797885
dtype: int64
Summary of Data:              Quantity                    InvoiceDate          Price  \
count  797885.000000                         797885  797885.000000
mean       12.602980  2011-01-02 13:17:34.141160704       3.702732
min    -80995.000000            2009-12-01 07:45:00       0.000000
25%         2.000000            2010-07-02 09:47:00       1.250000
50%         5.000000            2010-12-02 12:33:00       1.950000
75%        12.000000            2011-07-31 15:50:00       3.750000
max     80995.000000            2011-12-09 12:50:00   38970.000000
std       191.670371                            NaN      71.392549

         Customer ID
count  797885.000000
mean    15313.062777
min     12346.000000
25%     13964.000000
50%     15228.000000
75%     16788.000000
max     18287.000000
std      1696.466663
Removing Invoices that start with letters...
Number of records after removing invoices starting with letters:  779495
Removing prices that are negative or zero...
Number of records after removing negatively priced products:  779425
Removing Negative Quantities...
Number of records after removing Negative Quantity products:  779425
Summary of Data:             Quantity                    InvoiceDate          Price  \
count  779425.000000                         779425  779425.000000
mean       13.489370  2011-01-03 01:44:42.593475584       3.218488
min         1.000000            2009-12-01 07:45:00       0.001000
25%         2.000000            2010-07-02 14:39:00       1.250000
50%         6.000000            2010-12-02 14:09:00       1.950000
75%        12.000000            2011-08-01 13:44:00       3.750000
max     80995.000000            2011-12-09 12:50:00   10953.500000
std       145.855814                            NaN      29.676140

         Customer ID
count  779425.000000
mean    15320.360461
min     12346.000000
25%     13971.000000
50%     15247.000000
75%     16794.000000
max     18287.000000
std      1695.692775
```

After cleaning the data, we calculate Recency, Frequency and Monetary values. For recency, we substract the most recent date in dataset with the last transaction date of the customer in the dataset. For frequency, we count how many times has the customer made the interaction with the business. For monetary, we sum all the spending that customer has done in our business.
Once we get this information, we start assigning the R, F and M scores based on percentile model. See the model code on how we are using percentile to model the behavior of customers. Then we concatenate R,F and M scores to create a single RFM metric for segmentation. 

```bash
Calculating Recency, Frequency, Monetary Values and Merging Data...
Calculating RFM scores based on percentiles...
Single RFM Metric...
```

Now, we look at the distribution of Recency, Frequency and Monetary values of customers to visualize data distribution.
```bash
Plotting Data Distribution...
```
![distribution_plot](https://github.com/kkharel/K-Means-Clustering/assets/59852121/71ec924f-71cb-4df6-81bb-faedbd0b80ee)

From the plot, we can see that all the data are right-skewed(positive) and obviously not symmetric.

Skewness is a statistical measure that describes the asymmetry of a probability distribution. It tells us whether the data is concentrated more on one side of the mean than the other. A skewness value can be positive, negative, or zero: A positive skewness indicates that the right tail of the distribution is longer or fatter than the left, and the bulk of the values lie to the left of the mean. A negative skewness indicates that the left tail is longer or fatter than the right, and the bulk of the values lie to the right of the mean. A skewness value of zero means that the distribution on both sides of the mean is symmetrical.

Kurtosis is a statistical measure that describes the distribution of data in terms of the shape of its tails in relation to the normal distribution. Specifically, it indicates whether the data are heavy-tailed or light-tailed relative to a normal distribution. The kurtosis values can be classified into three types: Mesokurtic (Kurtosis = 0): The distribution has tails that are similar to the normal distribution. Leptokurtic (Kurtosis > 0): The distribution has heavier tails than the normal distribution, indicating more extreme values. Platykurtic (Kurtosis < 0): The distribution has lighter tails than the normal distribution, indicating fewer extreme values.

```bash
Checking Skewness and Kurtosis...
Skewness:
Recency       0.887198
Frequency    18.162730
Monetary     25.070190
dtype: float64
Kurtosis:
Recency       -0.476970
Frequency    523.518914
Monetary     830.167745
dtype: float64
```
From the skewness, we can see that all three variables have a positie skewness, suggesting that the distribution are right-skewed implying there are likely some extreme values (outliers) on the higher end of the distribution and majority of the values are concentrated on the lower end of the distribution. It could mean that there are few customers who have very recent transactions, highly frequent and/or have large monetary amounts while the majority of customers have lower values for these variables.

From the kurtosis, we can see that for Recency, the negative kurtosis value suggests that the distribution has lighter tails than a normal distribution, indicating fewer extreme values. For Frequency and Monetary, the positive kurtosis values indicate that the distributions have heavier tails than a normal distribution, suggesting more extreme values. This could imply that there are some customers with very high transaction frequency and monetary amounts, leading to the heavier tails in these distributions. In summary: Recency: Lighter tails (Platykurtic) Frequency and Monetary: Heavier tails (Leptokurtic)

The k-means algorithm is based on the mean of data points within clusters and its performance can be affected by the distribution of data. K means makes the assumption that clusters are spherical and equally sized, and it tries to minimize the variance within clusters which makes this algorithm sensitive to scaling and shape of the 
clusters.

Hence, we transform and scale the variables. We apply power transformation to Recency variable and log transformation to frequency and monetary variable.

```bash
Transforming Variables (Power & Log)...
```
Once, we apply transformation to the variables, lets look at the kde plot.

```bash
Plotting Transformed KDE...
```
![transformed_kde](https://github.com/kkharel/K-Means-Clustering/assets/59852121/ebfa5bf4-4338-48da-b2dc-93cd87e2e195)

From looking at the kde plot, we can see that Frequency and Monetary closely follows normal distribution as well as Recency but Recency exhibits bimodality which we need to handle differently.

Lets visualize the boxplot for transformed vairables before applying outlier handling
```bash
Boxplot of Transformed Variables...
```
![box_plot_transformed_before](https://github.com/kkharel/K-Means-Clustering/assets/59852121/d69ead8b-4a85-4e74-a343-bb1c0eced258)

From the boxplot of tranformed variables, it is evident that Frequency and Monetary have extreme values. There are various ways to handle the outliers, but we will use interquartile range method to replace the extreme values with bounds.

```bash
Replacing Outliers with Bounds...
```

Let's visualize the data with boxplot to see whether there are outliers present in the data or not after outlier handling.

```bash
Boxplot of Transformed Variables After Outlier Handling...
```

![box_plot_transformed](https://github.com/kkharel/K-Means-Clustering/assets/59852121/7377d0cb-cec4-402a-8e76-6c709356c09a)

The dataset looks good so far. Previously, we saw that power transforming the Recency exhibits bimodal density. We will apply the GMM model to separate them and scale each one afterwards I will create a new variable recency_bimodal to represent two distinct peaks in recency variable using gaussian mixture model.
Another caveat When working with days variables is that we need to make sure whether days should be represented as continuous variable or discrete variable. See below for variable treatment. Continuous: If "number of days" refers to a continuous quantity, such as the time elapsed between two events, it is considered continuous. For example, the time duration between two timestamps (measured with high precision) can be treated as a continuous variable. Discrete: On the other hand, if "number of days" is used to represent a count or a number of whole days (e.g., the number of days until an event occurs), then it is discrete. Discrete variables take on distinct, separate values and do not have values between them.

```bash
Separating bimodality of Recency using GMM...
```

Since, K means algorithm is distance based and is sensitive to larger and smaller values of variables, we scale them to bring all variables to same scale. Pay close attention in the code on how we scale the recency variable. The formula for scaling is as follows: x_scaled = x - x_min/x_max - x_min

```bash
Min-Max Scaling Variables...
```

After scaling the variables, let's compare the skewness and kurtosis between original variables, transformed variables and scaled variables. 

```bash
Comparision of Skewness and Kurtosis among variables...
Original Variables Skewness...
Skewness:
Recency       0.887198
Frequency    18.162730
Monetary     25.070190
dtype: float64
Transformed Variables Skewness...
Skewness:
power_Recency    0.087316
log_Frequency   -0.058030
log_Monetary     0.188739
dtype: float64
Scaled Variables Skewness...
Skewness:
scaled_Recency     -0.248486
scaled_Frequency   -0.058030
scaled_Monetary     0.188739
dtype: float64
Original Variables Kurtosis...
Kurtosis:
Recency       -0.476970
Frequency    523.518914
Monetary     830.167745
dtype: float64
Transformed Variables Kurtosis...
Kurtosis:
power_Recency   -1.286121
log_Frequency   -0.204255
log_Monetary    -0.145086
dtype: float64
Scaled Variables Kurtosis...
Kurtosis:
scaled_Recency     -0.914912
scaled_Frequency   -0.204255
scaled_Monetary    -0.145086
dtype: float64
```

The transformations and scaling have helped in reducing skewness, moving the distributions of the variables closer to symmetry. All the skewness values are relatively close to zero, with scaled_Recency being the most negatively skewed (-0.25), scaled_Frequency being close to zero (-0.03), and scaled_Monetary being slightly positively skewed (0.27). Generally, skewness values within the range of -0.5 to 0.5 are considered acceptable for assuming normality.

The transformations (power and logarithmic) and scaling have generally resulted in distributions with lighter tails compared to the original variables. Negative kurtosis values suggest a distribution with fewer extreme values or outliers than a normal distribution. The kurtosis values are also within a reasonable range. Scaled_Recency and Scaled_Frequency both have negative kurtosis values, indicating slightly platykurtic distributions, but the magnitudes are not extreme. Scaled_Monetary has a positive kurtosis value, indicating a slightly leptokurtic distribution, but again, the magnitude is not highly pronounced.

Now, we visualize the final density of the variables that we are going to use for KNN algorithm.

```bash
Plotting KDE of transformed variables...
```

![transformed_kde_2](https://github.com/kkharel/K-Means-Clustering/assets/59852121/b102e96f-a97a-4a87-80a5-dad325e69003)

Now I will map the RFM scores to its segments to generate new features and to evaluate later on how well did the knn algorithm learned the structure and relationship of the data.

```bash
Mapping Scores to Segments...
```

The mapping of scores are in the code itself, please take a look at it to understand the mapping structure. Once we map the scores to the segments, we one-hot encode te segments and use it as features for our k means algorithm.

```bash
Converting Segments into dummy variables...
```

Most of the heavy lifting is done besides writing algorithm from the scratch. Above section loads the data, cleans the data, transforms the data and explores different visualizations. Now, we select the features manually that goes into the k means algorithm to cluster the segments.

```bash
Selected features for KNN algorithm...
['Customer ID', 'scaled_RFM', 'bimodal_Recency', 'scaled_Recency', 'scaled_Frequency', 'scaled_Monetary', 'segment_A
boutToSleep', 'segment_AtRisk', 'segment_CannotLoseThem', 'segment_Champions', 'segment_Hibernating', 'segment_Lost'
, 'segment_LoyalCustomers', 'segment_NeedAttention', 'segment_PotentialLoyalist', 'segment_Promising', 'segment_Rece
ntCustomers']
```

Let's look at the correlation matrix of the features.

```bash
Plotting Correlation Heatmap...
```

![correlation_plot](https://github.com/kkharel/K-Means-Clustering/assets/59852121/af7c38b2-5a7c-45b6-b23d-1f4c899d7a3e)

From the correlation plot, we can see that some features are highly correlated, we will capture the most important patterns in the data and avoid redundancy. Since the algorithm is sensitive to scale and correlation of features, PCA may help us improve the performance of the model. We apply pca to our features and keep 90% of variability in the data

```bash
Applying PCA to remove correlation among features...
```
We will create a new dataframe to store the principal components after applying pca and then we will change the structure of the data to dictionary for the algorithm and check the first 3 records of the dataset and also perform Sanity Check so that we haven't make any mistakes so far.

```bash
Creating dataframe to store the principal components...
Converting dataframe to dictionary of values...
First 3 records of dictionary...
12346.0: [-0.4755740173852797, -0.39687357555408115, -0.053550049612977206, 0.6290246109423404, -0.14622237740762065
, -0.4831512148375629, -0.17453574993464746, 0.374691436097668, -0.3808253206643813]
12347.0: [1.0614989792265237, -0.4288626900216515, 0.3036127204476069, 0.052540525609238255, 0.08954268758778265, 0.
001741108308870603, -0.01291866834532743, -0.04785137742310192, 0.24865899420848558]
12348.0: [0.2635960027247667, 0.4043083407063475, -0.1705938582702758, -0.06460244895084082, -0.04269566595068019, -
0.2162098825461374, 0.8498665971148607, -0.2605029859061856, -0.48753521396875876]
Sanity check of data...
Number of keys: 5878
Number of values: 5878
```

To choose the optimal K value, we use an elbow method but we skip this for now since we know that we have 11 segments and we have defined it manually above. Not needed we have domain knowledge of 11 segments. In case, if we want to try this method as well, then code is present in the model as comments which will help us find the optimal k value. We create a copy of segment variable so that we can use it to evaluate the model performace during model evaluation step

```bash
Creating check segment variable to evaluate the performance of algorithm...
```

Running the k means algorithm with Euclidean distance metric. Notice how I have handled the case when there are no points assigned to clusters during first couple iterations.

```
Running K-Means Algorithm with Euclidean distance metric...
Converged at iteration 8
```

From the tableau dashboard, I know that there are 1268 champions in the dataset. So I try to sanity check the numbers to see the learning of the algorithm. The difference between the numbers is very miniscule and it comes from handling and capping the outliers above.

```bash
Sanity Check Champions...
Number of values: 1273
```

Now, we visualize the result of the clustering algorithm. This is an interactive plot so we need to host this using html file. Please see the link below for final clustering using the algorithm.

```bash
Plotting Clusters...
```

[Link to Plotly Plot](https://kkharel.github.io/K-Means-Clustering/index.html)


Finally we evaluate how well our model is doing by looking at various qualitative and quantitative methods and provide recommendation for business to make decisions.

```bash
Model Evaluation...
    Cluster Label      check_segment  Count
0               0        Hibernating    241
1               1          Champions   1273
2               2        Hibernating    276
3               3       AboutToSleep    543
4               4      NeedAttention    198
5               4  PotentialLoyalist    435
6               4          Promising    138
7               4    RecentCustomers    356
8               5             AtRisk    230
9               6     LoyalCustomers    562
10              7               Lost    795
11              8     CannotLoseThem    139
12              8      NeedAttention    215
13              8  PotentialLoyalist      8
14              9             AtRisk    219
15             10        Hibernating    250
Silhouette Score: 0.5739293624937142
Inertia: 1317.044643949626
Explained Variance Ratio: [0.36194616 0.11035282 0.10355967 0.08092097 0.07298444 0.0619178
 0.05206921 0.0494782  0.04511895]
check_segment  AboutToSleep  AtRisk  CannotLoseThem  Champions  Hibernating  \
Cluster Label
0                         0       0               0          0          241
1                         0       0               0       1273            0
2                         0       0               0          0          276
3                       543       0               0          0            0
4                         0       0               0          0            0
5                         0     230               0          0            0
6                         0       0               0          0            0
7                         0       0               0          0            0
8                         0       0             139          0            0
9                         0     219               0          0            0
10                        0       0               0          0          250

check_segment  Lost  LoyalCustomers  NeedAttention  PotentialLoyalist  \
Cluster Label
0                 0               0              0                  0
1                 0               0              0                  0
2                 0               0              0                  0
3                 0               0              0                  0
4                 0               0            198                435
5                 0               0              0                  0
6                 0             562              0                  0
7               795               0              0                  0
8                 0               0            215                  8
9                 0               0              0                  0
10                0               0              0                  0

check_segment  Promising  RecentCustomers
Cluster Label
0                      0                0
1                      0                0
2                      0                0
3                      0                0
4                    138              356
5                      0                0
6                      0                0
7                      0                0
8                      0                0
9                      0                0
10                     0                0
```

Silhouette Score( between -1 and 1) - Measures how well separated the clusters are where higher score represents well separated clusters. Inertia (within cluster sum of squares): Measures how compact the clusters are. Lower values are better.

From inertia silhouette score, K-means clustering has produced clusters that are both compact (low inertia) and well-separated (high silhouette score). This combination indicates that the algorithm has effectively grouped similar data points together while keeping the clusters distinct from each other.

Clusters 3, 6, and 8 appears to have a significant number of instances, suggesting they may represent distinct and well-defined customer groups. Clusters 3, 6, and 8 have clear dominant segments ('AboutToSleep', 'LoyalCustomers', and 'CannotLoseThem', respectively), indicating a focused and specific customer profile for these clusters.

Clusters 0, 1, 2, 4, 5, 7, 9, and 10, having few or no instances in certain segments, suggest that these clusters are more homogenous with respect to the segmentation variable. This can be advantageous for targeted strategies since the behavior of customers within these clusters is more uniform.

Cluster 4 still stands out as having a diverse set of segments ('PotentialLoyalist', 'NeedAttention', 'Promising', 'RecentCustomers'). This diversity may indicate a cluster with varied customer behaviors, and strategies for this cluster might need to be more flexible.

The business recommendation we have is For clusters with lack of variety, targeted strategies can be more straightforward and tailored to the dominant segment. For example, Cluster 6 ('LoyalCustomers') might be approached with loyalty reward programs. 

Clusters with less variety may be easier to understand and manage. For instance, if Cluster 8 ('CannotLoseThem') primarily consists of customers who are reluctant to switch brands, marketing efforts can focus on maintaining their satisfaction.

Given the diversity in Cluster 4, a more nuanced and adaptable strategy may be needed. This cluster might benefit from marketing campaigns that can appeal to different customer preferences.

In summary, the lack of variety in certain clusters is advantageous for segmentation, simplifying the development of targeted strategies. However, it's crucial to strike a balance, as some diversity (like in Cluster 4) might be desirable for capturing a broader range of customer behaviors. 

Dataset Citation:

@misc{misc_online_retail_ii_502,
  author       = {Chen,Daqing},
  title        = {{Online Retail II}},
  year         = {2019},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5CG6D}
}
