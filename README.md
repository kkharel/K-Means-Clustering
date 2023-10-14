# K-Means-Clustering
Use Cases ex - Market Segmentation, Social Network Analysis, Search result groupings, Medical imaging, image segmentation, anomaly detection, Recommendation systems etc...
K means complexity O(n).
Types of Clustering: Centroid Based Clustering, Density Based Clustering, Distribution Based Clustering, Hierarchical Clustering
  
Steps to cluster data: Prepare data, Create similarity Metric, Run Clustering Algorithm, Interpret results and adjust clustering

RFM Analysis of Customers 

# Check the skewness of data

# Skewness is a statistical measure that describes the asymmetry of a probability distribution. 
# It tells us whether the data is concentrated more on one side of the mean than the other.
# A skewness value can be positive, negative, or zero:
# A positive skewness indicates that the right tail of the distribution is longer or fatter 
# than the left, and the bulk of the values lie to the left of the mean.
# A negative skewness indicates that the left tail is longer or fatter than the right, 
# and the bulk of the values lie to the right of the mean.
# A skewness value of zero means that the distribution on both sides of the mean is symmetrical.

# From the skewness, we can see that all three variables have a positie skewness, suggesting
# that the distribution are right-skewed implying there are likely some extreme values (outliers)
# on the higher end of the distribution and majority of the values are concentrated on the lower
# end of the distribution. It could mean that there are few customers who have very recent
# transactions, highly frequent and/or have large monetary amounts while the majority of
# customers have lower values for these variables


# Kurtosis is a statistical measure that describes the distribution of data in terms of 
# the shape of its tails in relation to the normal distribution. 
# Specifically, it indicates whether the data are heavy-tailed or light-tailed 
# relative to a normal distribution.

# The kurtosis values can be classified into three types:

# Mesokurtic (Kurtosis = 0): 
# The distribution has tails that are similar to the normal distribution.

# Leptokurtic (Kurtosis > 0): 
# The distribution has heavier tails than the normal distribution, indicating more extreme values.

# Platykurtic (Kurtosis < 0): 
# The distribution has lighter tails than the normal distribution, indicating fewer extreme values.

# From our results, we can see that

# For Recency, the negative kurtosis value suggests that the distribution has lighter 
# tails than a normal distribution, indicating fewer extreme values.

# For Frequency and Monetary, the positive kurtosis values indicate that the distributions 
# have heavier tails than a normal distribution, suggesting more extreme values. 
# This could imply that there are some customers with very high transaction 
# frequency and monetary amounts, leading to the heavier tails in these distributions.

# In summary:
# Recency: Lighter tails (Platykurtic)
# Frequency and Monetary: Heavier tails (Leptokurtic)

# The k-means algorithm is based on the mean of data points within clusters and its
# performance can be affected by the distribution of data. K means makes the assumption
# that clusters are spherical and equally sized, and it tries to minimize the variance
# within clusters which makes this algorithm sensitive to scaling and shape of the 
# clusters.

# Hence, we transform and scale the variables. We apply power transformation to Recency
# variable and log transformation to frequency and monetary variable.


# From looking at the kde plot, we can see that Frequency and Monetary closely follows
# normal distribution as well as Recency but Recency exhibits bimodality.

# Now, let's look at boxplot of above transformed variables

# From the boxplot of tranformed variables, it is evident that Frequency and Monetary
# have extreme values. There are various ways to handle the outliers, but we will use
# interquartile range method to replace the extreme values with bounds.

# Previously, we saw that power transforming the Recency exhibits bimodal density. We will
# apply the GMM model to separate them and scale each one afterwards
# I will create a new variable recency_bimodal to represent two distinct peaks in recency variable using
# gaussian mixture model.
# When working with days variables, we need to make sure whether days should be represented
# as continuous variable or discrete variable. See below for variable treatment.

# Continuous: If "number of days" refers to a continuous quantity, such as the time 
# elapsed between two events, it is considered continuous. 
# For example, the time duration between two timestamps (measured with high precision) 
# can be treated as a continuous variable.

# Discrete: On the other hand, if "number of days" is used to represent a count or a number
# of whole days (e.g., the number of days until an event occurs), 
# then it is discrete. Discrete variables take on distinct, 
# separate values and do not have values between them.

# x_scaled = x - x_min/x_max - x_min
# Now we will scale our tranformed variables to same scale for knn algorithm

# Scaling Recency, Frequency and Monetary Values. Notice on how I tranformed the 
# recency variable to take into account the bimodal nature.

# Now, let's compare the skewness and kurtosis between original variables, transformed variables
# and scaled variables

# The transformations and scaling have helped in reducing skewness, moving the 
# distributions of the variables closer to symmetry.
# All the skewness values are relatively close to zero, with scaled_Recency 
# being the most negatively skewed (-0.25), scaled_Frequency being close to zero (-0.03), 
# and scaled_Monetary being slightly positively skewed (0.27). 
# Generally, skewness values within the range of -0.5 to 0.5 are considered 
# acceptable for assuming normality.

#Kurtosis:
# The transformations (power and logarithmic) and scaling have generally resulted in 
# distributions with lighter tails compared to the original variables. 
# Negative kurtosis values suggest a distribution with fewer extreme values or 
# outliers than a normal distribution.
#The kurtosis values are also within a reasonable range. 
# scaled_Recency and scaled_Frequency both have negative kurtosis values, 
# indicating slightly platykurtic distributions, but the magnitudes are not extreme. 
# scaled_Monetary has a positive kurtosis value, indicating a slightly leptokurtic distribution, 
# but again, the magnitude is not highly pronounced.

# Now I will map the RFM scores to its segments so that we can assess how well
# did the knn algorithm learned the structure and relationship of the data.

# Mapping RFM scores to segments

# Converting segment into dummy variables

# Our data is ready for knn algorithm. Now, we will select the variables that goes into
# our algorithm

# Let's look at the correlation matrix of the features

# From the correlation plot, we can see that some features are highly correlated, 
# we will capture the most important patterns in the data and avoid redundancy. 
# Since the algorithm is sensitive to scale and correlation of features, 
# PCA may help us improve the performance of the model.

# We apply pca to our features and keep 90% of variability in the data

# We will change the structure of the data to dictionary for the algorithm

# checking the first 5 records of the dataset

# Sanity Check

# To choose the optimal K value, we use an elbow method but we skip this for now
# since we know that we have 11 segments and we have defined it manually above
# Not needed we have domain knowledge of 11 segments.
# In case, if we want to try this method as well, then below code will help us
# find the optimal k value

# Now, we visualize the result of the clustering algorithm

# Model Evaluation

# Silhouette Score( between -1 and 1) - Measures how well separated the clusters are where
# higher score represents well separated clusters.

# Inertia (within cluster sum of squares): Measures how compact the clusters are. Lower
# values are better

# From inertia silhouette score, K-means clustering has produced clusters that are both 
# compact (low inertia) and well-separated (high silhouette score).
# This combination indicates that the algorithm has effectively grouped similar data points 
# together while keeping the clusters distinct from each other.

# Clusters 3, 6, and 8 appears to have a significant number of instances,
# suggesting they may represent distinct and well-defined customer groups.
# Clusters 3, 6, and 8 have clear dominant segments 
# ('AboutToSleep', 'LoyalCustomers', and 'CannotLoseThem', respectively), 
# indicating a focused and specific customer profile for these clusters.

# Clusters 0, 1, 2, 4, 5, 7, 9, and 10, having few or no instances in certain segments, 
# suggest that these clusters are more homogenous with respect to the 
# segmentation variable. This can be advantageous for targeted strategies since the 
# behavior of customers within these clusters is more uniform.

# Cluster 4 still stands out as having a diverse set of segments 
# ('PotentialLoyalist', 'NeedAttention', 'Promising', 'RecentCustomers'). 
# This diversity may indicate a cluster with varied customer behaviors, 
# and strategies for this cluster might need to be more flexible.

### Recommendations:

# Targeted Strategies

# For clusters with lack of variety, targeted strategies can be more straightforward 
# and tailored to the dominant segment. 
# For example, Cluster 6 ('LoyalCustomers') might be approached with loyalty reward programs.

# Clusters with less variety may be easier to understand and manage.
# For instance, if Cluster 8 ('CannotLoseThem') primarily consists of customers 
# who are reluctant to switch brands, marketing efforts can focus on maintaining 
# their satisfaction.

# Given the diversity in Cluster 4, a more nuanced and adaptable strategy may be needed. 
# This cluster might benefit from marketing campaigns that can appeal to different customer 
# preferences.

# In summary, the lack of variety in certain clusters is advantageous for segmentation, 
# simplifying the development of targeted strategies. 
# However, it's crucial to strike a balance, as some diversity 
# (like in Cluster 4) might be desirable for capturing a broader range of customer behaviors. 
