import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_score


os.chdir("C:/Users/kkhar/OneDrive/Desktop/K-Means-Clustering")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

def readexcel(excelfile):
  xls = pd.ExcelFile(excelfile)
  sheet_names = xls.sheet_names
  num_sheets = len(sheet_names)
  print("Sheet Names:", sheet_names)
  print("Number of Sheets:", num_sheets)
  
  data = {}
  
  for sheet_index in range(len(sheet_names)):
    data[sheet_index] = pd.read_excel(excelfile, sheet_name = sheet_index)
  
  df = pd.concat(data.values(), ignore_index = True)
  return df

def starts_with_letter(string):
  string = str(string)
  return string[0].isalpha()

def datacleaning(df):
  print(df.info())
  null_mask = df.isnull().any(axis=1)
  null_rows = df[null_mask]
  print("Count of Null Rows before Removal: ", len(null_rows))
  df = df.dropna()
  print("Count of Nulls in Columns after Removal: ", df.isnull().sum())
  print("Number of Records after Null Removal: ", len(df))
  print("Number of records before removing duplicates: ", df.count())
  df = df.drop_duplicates()
  print("Number of records after removing duplicates: ", df.count())
  print("Summary of Data: ", df.describe())
  print("Removing Invoices that start with letters...")
  df = df[~df['Invoice'].apply(starts_with_letter)]
  print("Number of records after removing invoices starting with letters: ", len(df))
  print("Removing prices that are negative or zero...")
  df = df[~(df['Price'] <= 0)]
  print("Number of records after removing negatively priced products: ", len(df))
  print("Removing Negative Quantities...")
  df = df[~(df['Quantity'] < 0)]
  print("Number of records after removing Negative Quantity products: ", len(df))
  print("Summary of Data:", df.describe())
  return df  
  
def get_recency(df):
  final_data = df.copy()
  dataset_max = pd.to_datetime(final_data['InvoiceDate'].max())
  customer_max = final_data.groupby('Customer ID', as_index = False)['InvoiceDate'].max()
  customer_max.columns = ['Customer ID', 'Latest_Invoice_Date']
  customer_max['Latest_Invoice_Date'] = pd.to_datetime(customer_max['Latest_Invoice_Date'])
  customer_max['Recency'] = customer_max.Latest_Invoice_Date.apply(lambda x: (dataset_max - x).days) 
  customer_max = customer_max.drop(['Latest_Invoice_Date'], axis = 1)
  return customer_max

def get_frequency(df):
  final_data = df.copy()
  Item_Count = final_data.groupby(['Customer ID'], as_index = False).agg({'Invoice': lambda x: len(x)})
  Item_Count.columns = ['Customer ID', 'Frequency']
  return Item_Count

def get_monetary(df):
  final_data = df.copy()
  final_data['Sales'] = final_data['Price']*final_data['Quantity']
  Total_Sales = final_data.groupby('Customer ID')['Sales'].sum().reset_index()
  Total_Sales.columns = ['Customer ID', 'Monetary']
  return Total_Sales

def mergedata(df):
  recency = get_recency(df = df)
  recency = recency.reset_index(drop = True)
  frequency = get_frequency(df = df)
  frequency = frequency.reset_index(drop = True)
  monetary = get_monetary(df = df)
  monetary = monetary.reset_index(drop = True)
  merge1 = pd.merge(recency, frequency, on = 'Customer ID')
  merge2 = pd.merge(merge1, monetary, on = 'Customer ID')
  return merge2


def calculate_rfm_scores(dataframe, r_col, f_col, m_col):
  #dataframe.sort_values(by=[r_col, f_col, m_col], inplace=True)  # Sort the DataFrame by columns
  r_quantiles = dataframe[r_col].quantile([0.2, 0.4, 0.6, 0.8])
  f_quantiles = dataframe[f_col].quantile([0.2, 0.4, 0.6, 0.8])
  m_quantiles = dataframe[m_col].quantile([0.2, 0.4, 0.6, 0.8])

  def R_score(recency):
    if recency <= r_quantiles.iloc[0]:
      return 5
    elif recency > r_quantiles.iloc[0] and recency <= r_quantiles.iloc[1]:
      return 4
    elif recency > r_quantiles.iloc[1] and recency <= r_quantiles.iloc[2]:
      return 3
    elif recency > r_quantiles.iloc[2] and recency <= r_quantiles.iloc[3]:
      return 2
    else:
      return 1

  def F_score(frequency):
    if frequency <= f_quantiles.iloc[0]:
      return 1
    elif frequency > f_quantiles.iloc[0] and frequency <= f_quantiles.iloc[1]:
      return 2
    elif frequency > f_quantiles.iloc[1] and frequency <= f_quantiles.iloc[2]:
      return 3
    elif frequency > f_quantiles.iloc[2] and frequency <= f_quantiles.iloc[3]:
      return 4
    else:
      return 5

  def M_score(monetary):
    if monetary <= m_quantiles.iloc[0]:
      return 1
    elif monetary > m_quantiles.iloc[0] and monetary <= m_quantiles.iloc[1]:
      return 2
    elif monetary > m_quantiles.iloc[1] and monetary <= m_quantiles.iloc[2]:
      return 3
    elif monetary > m_quantiles.iloc[2] and monetary <= m_quantiles.iloc[3]:
      return 4
    else:
      return 5

  dataframe['R'] = dataframe[r_col].apply(R_score)
  dataframe['F'] = dataframe[f_col].apply(F_score)
  dataframe['M'] = dataframe[m_col].apply(M_score)

  return dataframe[['R', 'F', 'M']]


def plot_combined_distribution(data, columns, xlims=None):
  num_plots = len(columns)
  fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(15, 5))

  for i, column in enumerate(columns):
    sns.histplot(data[column], ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')
    if xlims and xlims[i]:
      axes[i].set_xlim(xlims[i])

  plt.tight_layout()
  plt.savefig('distribution_plot.jpg', format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def check_skew(df, columns):
  skewness = df[columns].skew()
  print("Skewness: ")
  print(skewness)
  return skewness

def check_kurtosis(df, columns):
  kurtosis = df[columns].kurtosis()
  print("Kurtosis: ")
  print(kurtosis)
  return kurtosis

def power_transform(df, columns, power):
  for col in columns:
    df[f'power_{col}'] = (df[columns]+1)**power
  return df 

def log_transform(df, columns):
  for col in columns:
    df[f'log_{col}'] = np.log(df[col] + 1)
  return df

def plot_transformed_kde(df, transformed_cols):
  fig, ax = plt.subplots()

  for col in transformed_cols:
    sns.kdeplot(df[col], label=f'{col}', ax=ax)

  ax.legend()
  plt.xlabel('')
  plt.savefig('transformed_kde.jpg', format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def create_boxplot_multi_variables(df, variable_list):
  plt.figure(figsize=(15, 12))
  plt.boxplot([df[var] for var in variable_list], labels=variable_list)
  plt.ylabel('Values')
  plt.title('Box Plot of Variables')
  plt.savefig('box_plot_transformed.jpg', format = 'jpg', dpi = 300, bbox_inches = 'tight')
  plt.show()

def replace_outliers_with_bounds_multi_variables(df, variable_list, percentile_low=25, percentile_high=75):
  for var in variable_list:
    data_series = df[var]
    Q1 = np.percentile(data_series, percentile_low)
    Q3 = np.percentile(data_series, percentile_high)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[var] = np.where((df[var] < lower_bound), lower_bound, df[var])
    df[var] = np.where((df[var] > upper_bound), upper_bound, df[var])
  return df

def separate_recency(df, columns, components, randomstate):
  gmm = GaussianMixture(n_components=components, random_state=randomstate)
  for cols in columns:
    df[f'bimodal_{cols}'] = gmm.fit_predict(pd.to_numeric(df[columns[0]]).to_numpy().reshape(-1, 1))
  return df

def minMaxScaler(numcol):
  minx = np.min(numcol)
  maxx = np.max(numcol)
  scaled_col = (numcol - minx) / (maxx - minx)
  return scaled_col


def rfm_score_to_label(score):
  rfm_mapping = {
    555: "Champions",
    554: "Champions",
    553: "LoyalCustomers",
    552: "PotentialLoyalist",
    551: "PotentialLoyalist",
    545: "Champions",
    544: "Champions",
    543: "LoyalCustomers",
    542: "PotentialLoyalist",
    541: "PotentialLoyalist",
    535: "PotentialLoyalist",
    534: "PotentialLoyalist",
    533: "PotentialLoyalist",
    532: "PotentialLoyalist",
    531: "PotentialLoyalist",
    525: "Promising",
    524: "Promising",
    523: "Promising",
    522: "RecentCustomers",
    521: "RecentCustomers",
    515: "Promising",
    514: "Promising",
    513: "Promising",
    512: "RecentCustomers",
    511: "RecentCustomers",
    455: "Champions",
    454: "Champions",
    453: "LoyalCustomers",
    452: "PotentialLoyalist",
    451: "PotentialLoyalist",
    445: "Champions",
    444: "Champions",
    443: "LoyalCustomers",
    442: "PotentialLoyalist",
    441: "PotentialLoyalist",
    435: "PotentialLoyalist",
    434: "PotentialLoyalist",
    433: "PotentialLoyalist",
    432: "PotentialLoyalist",
    431: "PotentialLoyalist",
    425: "Promising",
    424: "Promising",
    423: "Promising",
    422: "RecentCustomers",
    421: "RecentCustomers",
    415: "Promising",
    414: "Promising",
    413: "Promising",
    412: "RecentCustomers",
    411: "RecentCustomers",
    355: "LoyalCustomers",
    354: "LoyalCustomers",
    353: "PotentialLoyalist",
    352: "PotentialLoyalist",
    351: "PotentialLoyalist",
    345: "LoyalCustomers",
    344: "LoyalCustomers",
    343: "NeedAttention",
    342: "NeedAttention",
    341: "NeedAttention",
    335: "NeedAttention",
    334: "NeedAttention",
    333: "NeedAttention",
    332: "NeedAttention",
    331: "AboutToSleep",
    325: "NeedAttention",
    324: "NeedAttention",
    323: "NeedAttention",
    322: "AboutToSleep",
    321: "AboutToSleep",
    315: "NeedAttention",
    314: "NeedAttention",
    313: "AboutToSleep",
    312: "AboutToSleep",
    311: "AboutToSleep",
    255: "AtRisk",
    254: "AtRisk",
    253: "AtRisk",
    252: "AboutToSleep",
    251: "AboutToSleep",
    245: "AtRisk",
    244: "AtRisk",
    243: "AtRisk",
    242: "AboutToSleep",
    241: "AboutToSleep",
    235: "AtRisk",
    234: "AtRisk",
    233: "AtRisk",
    232: "AboutToSleep",
    231: "AboutToSleep",
    225: "AtRisk",
    224: "AtRisk",
    223: "Hibernating",
    222: "Hibernating",
    221: "AboutToSleep",
    215: "AtRisk",
    214: "AtRisk",
    213: "Hibernating",
    212: "Hibernating",
    211: "Hibernating",
    155: "CannotLoseThem",
    154: "CannotLoseThem",
    153: "Hibernating",
    152: "Hibernating",
    151: "Lost",
    145: "CannotLoseThem",
    144: "CannotLoseThem",
    143: "Hibernating",
    142: "Hibernating",
    141: "Lost",
    135: "CannotLoseThem",
    134: "CannotLoseThem",
    133: "Hibernating",
    132: "Hibernating",
    131: "Lost",
    125: "CannotLoseThem",
    124: "CannotLoseThem",
    123: "Hibernating",
    122: "Lost",
    121: "Lost",
    115: "CannotLoseThem",
    114: "CannotLoseThem",
    113: "Lost",
    112: "Lost",
    111: "Lost",
  }
  return rfm_mapping.get(score, "Unknown")

def correlation_heatmap(dataframe, save_path='correlation_plot.jpg'):
  correlation_matrix = dataframe.corr()
  plt.figure(figsize=(10, 6))
  heatmap = sns.heatmap(correlation_matrix, annot=True, fmt="0.2f", cmap="coolwarm_r", annot_kws={"size": 8})
  heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, horizontalalignment="right", fontsize=12)
  heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, horizontalalignment="right", fontsize=12)
  plt.title("Correlation Heatmap of Features", fontsize=12)
  plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
  plt.show()


#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

#X = principal_components
#k_values = range(1, 21)
#inertia = []
#for k in k_values:
#  kmeans = KMeans(n_clusters=k, random_state=11, n_init='auto')
#  kmeans.fit(X)
#  inertia.append(kmeans.inertia_)
#plt.clf()
#plt.plot(k_values, inertia, marker='o')
#plt.xlabel('Number of Clusters (k)')
#plt.ylabel('Inertia')
#plt.title('Elbow Method for Optimal k')
#plt.savefig('elbow_plot.jpg', format = 'jpg', dpi = 300, bbox_inches = 'tight')
#plt.show()


# Algorithm

import random
import copy

def k_means(cluster_data_dict, num_clusters=11, max_iterations=20):
  # Selecting random initial centroids
  random.seed(42)
  initial_centroid_customerID = random.sample(list(cluster_data_dict.keys()), num_clusters)
  
  # Initializing centroids
  centroids = {f'cluster {ik} Centroids': cluster_data_dict[initial_centroid_customerID[ik]] for ik in range(num_clusters)}
  
  # Getting the number of features per user
  num_features_per_user = len(list(cluster_data_dict.values())[0])
  
  # Main loop for K-means algorithm
  
  # Previous centroids to check for convergence
  prev_centroids = copy.deepcopy(centroids)
  
  for iteration in range(max_iterations):
    # Initializing clusters at the beginning of each iteration
    clusters = {f'cluster {ik} CustomerID': [] for ik in range(num_clusters)}
    # Calculating distances
    distances = {f'centroid {ik} distance': {user: sum([(centroids[f'cluster {ik} Centroids'][feature] - cluster_data_dict[user][feature])**2 for feature in range(num_features_per_user)]) for user in cluster_data_dict} for ik in range(num_clusters)}
    # Assigning each user to the nearest centroid
    for user in cluster_data_dict:
      temp_distance = [distances[f'centroid {ik} distance'][user] for ik in range(num_clusters)]
      clusters[f'cluster {temp_distance.index(min(temp_distance))} CustomerID'].append(user)
    # Updating centroids
    for ik in range(num_clusters):
      mean_value = [0] * num_features_per_user
      cluster_size = len(clusters[f'cluster {ik} CustomerID'])
      if cluster_size != 0:
        for user in clusters[f'cluster {ik} CustomerID']:
          mean_value = [mean_value[feature] + cluster_data_dict[user][feature] for feature in range(num_features_per_user)]
        centroids[f'cluster {ik} Centroids'] = [mean_value[feature] / cluster_size for feature in range(num_features_per_user)]
      else:
        # If a cluster is empty, assign a random point as its centroid
        centroids[f'cluster {ik} Centroids'] = cluster_data_dict[random.choice(list(cluster_data_dict.keys()))]
  
    # Checking for convergence
    if prev_centroids == centroids:
      print(f"Converged at iteration {iteration + 1}")
      break
  
    # Updating previous centroids for the next iteration
    prev_centroids = copy.deepcopy(centroids)
  
    # print(f"Iteration {iteration + 1}: Clusters = {clusters}")
  return clusters, centroids


def plot_clusters_2d(data_dict, centroids, clusters):
  fig = go.Figure()
  colormap = px.colors.qualitative.Set1
  for i, centroid_key in enumerate(centroids):
    if isinstance(centroids[centroid_key], list):
      centroid_location = centroids[centroid_key]
    else:
      centroid_location = centroids[centroid_key].location
    fig.add_trace(go.Scatter(x=[centroid_location[0]], y=[centroid_location[1]], mode='markers', marker=dict(size=8, color='black', symbol='x'), hoverinfo='text', text=[f'Centroid of {centroid_key}'], name=f'Centroids {i}'))
  for i, cluster_key in enumerate(clusters):
    cluster_points = clusters[cluster_key]
    cluster_data = {k: data_dict[k] for k in cluster_points}
    x = [item[0] for item in cluster_data.values()]
    y = [item[1] for item in cluster_data.values()]
    customer_ids = list(cluster_data.keys())
    hover_text = [ f'Customer ID: {customer_id}<br>Count: {len(cluster_data)}<br>RFM: {df.loc[df["Customer ID"] == customer_id, "RFM"].values[0]}<br>Segment: {df.loc[df["Customer ID"] == customer_id, "check_segment"].values[0]}' for customer_id in customer_ids]
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=5, opacity=0.7, color=colormap[i % len(colormap)]), hoverinfo='text', text=hover_text, name=f'Cluster {i} Points'))
  fig.update_layout(xaxis_title='Principal Component 0', yaxis_title='Principal Component 1', title='K-Means Clustering')
  fig.write_html('clusters.html')
  fig.show()

def evaluate_model(df, clusters, centroids, num_clusters, principal_components, cluster_data_dict, num_features_per_user, pca):
  df['Cluster Label'] = -1 
  for ik in range(num_clusters):
    df.loc[df['Customer ID'].isin(clusters[f'cluster {ik} CustomerID']), 'Cluster Label'] = ik

  result = df.groupby(['Cluster Label', 'check_segment'])['Customer ID'].count().reset_index(name='Count')
  print(result)

  silhouette_avg = silhouette_score(principal_components, df['Cluster Label'])
  print(f"Silhouette Score: {silhouette_avg}")

  inertia = 0.0

  for ik in range(num_clusters):
    cluster_points = clusters[f'cluster {ik} CustomerID']
    cluster_center = centroids[f'cluster {ik} Centroids']
    for user in cluster_points:
      user_data = cluster_data_dict[user]
      inertia += sum([(user_data[feature] - cluster_center[feature])**2 for feature in range(num_features_per_user)])

  print(f"Inertia: {inertia}")

  explained_variance_ratio = pca.explained_variance_ratio_
  print(f"Explained Variance Ratio: {explained_variance_ratio}")

  crosstab_result = pd.crosstab(df['Cluster Label'], df['check_segment'])
  print(crosstab_result)


if __name__ == "__main__":
  print("Loading Excel File...")
  df = readexcel(excelfile="retail.xlsx")
  print("Cleaning Data...")
  df = datacleaning(df=df)
  print("Calculating Recency, Frequency, Monetary Values and Merging Data...")
  df = mergedata(df = df)
  print("Calculating RFM scores based on percentiles...")
  rfm_scores = calculate_rfm_scores(df, 'Recency', 'Frequency', 'Monetary')
  print("Single RFM Metric...")
  df['RFM'] = df['R'].astype(str)+df['F'].astype(str) + df['M'].astype(str)
  print("Plotting Data Distribution...")
  columns_to_plot = ['Recency', 'Frequency', 'Monetary']
  xlims = [(None, None), (0, 900), (0, 10000)]
  plot_combined_distribution(df, columns_to_plot, xlims)
  print("Checking Skewness and Kurtosis...")
  check_skew(df, columns = ['Recency', 'Frequency', 'Monetary'])
  check_kurtosis(df, columns = ['Recency', 'Frequency', 'Monetary'])
  print("Transforming Variables (Power & Log)...")
  df = power_transform(df=df, columns = ['Recency'], power = 0.32)
  df = log_transform(df, columns = ['Frequency', 'Monetary'])
  print("Plotting Transformed KDE...")
  plot_transformed_kde(df, ['power_Recency', 'log_Frequency', 'log_Monetary'])
  print("Boxplot of Transformed Variables...")
  create_boxplot_multi_variables(df, ['power_Recency', 'log_Frequency', 'log_Monetary'])
  print("Replacing Outliers with Bounds...")
  df = replace_outliers_with_bounds_multi_variables(df, ['log_Frequency', 'power_Recency', 'log_Monetary'])
  print("Boxplot of Transformed Variables After Outlier Handling...")
  create_boxplot_multi_variables(df, ['power_Recency', 'log_Frequency', 'log_Monetary'])
  print("Separating bimodality of Recency using GMM...")
  df = separate_recency(df=df, columns=['Recency'], components=2, randomstate=13)
  print("Min-Max Scaling Variables")
  df['scaled_Recency'] = df.groupby('bimodal_Recency')['power_Recency'].transform(minMaxScaler)
  df['scaled_Frequency'] = minMaxScaler(df['log_Frequency'])
  df['scaled_Monetary']  = minMaxScaler(df['log_Monetary'])
  df['scaled_RFM'] = minMaxScaler(df['RFM'].astype(int))
  print("Comparision of Skewness and Kurtosis among variables...")
  print("Original Variables Skewness...")
  check_skew(df, columns = ['Recency', 'Frequency', 'Monetary'])
  print("Transformed Variables Skewness...")
  check_skew(df, columns = ['power_Recency', 'log_Frequency', 'log_Monetary'])
  print("Scaled Variables Skewness...")
  check_skew(df, columns = ['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary'])
  print("Original Variables Kurtosis...")
  check_kurtosis(df, columns = ['Recency', 'Frequency', 'Monetary'])
  print("Transformed Variables Kurtosis...")
  check_kurtosis(df, columns = ['power_Recency', 'log_Frequency', 'log_Monetary'])
  print("Scaled Variables Kurtosis...")
  check_kurtosis(df, columns = ['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary'])
  print("Plotting KDE of transformed variables...")
  plot_transformed_kde(df, ['scaled_Recency', 'scaled_Frequency', 'scaled_Monetary'])
  print("Mapping Scores to Segments...")
  df['segment'] = df['RFM'].astype(int).apply(rfm_score_to_label)
  print("Converting Segments into dummy variables...")
  df = pd.get_dummies(df, columns=['segment'])
  df[df.columns[df.columns.str.startswith('segment_')]] = df[df.columns[df.columns.str.startswith('segment_')]].astype(int)
  print("Selected features for KNN algorithm...")
  cols = ['Customer ID', 'scaled_RFM', 'bimodal_Recency', 'scaled_Recency', 'scaled_Frequency', 'scaled_Monetary',  'segment_AboutToSleep', 'segment_AtRisk', 'segment_CannotLoseThem', 'segment_Champions', 'segment_Hibernating', 'segment_Lost', 'segment_LoyalCustomers', 'segment_NeedAttention', 'segment_PotentialLoyalist', 'segment_Promising', 'segment_RecentCustomers']
  print(cols)
  cluster_data = df[cols].astype(float)
  print("Plotting Correlation Heatmap...")
  correlation_heatmap(cluster_data)
  print("Applying PCA to remove correlation among features...")
  pca = PCA(n_components = 0.90) # keep 90% variability of data
  principal_components = pca.fit_transform(cluster_data.loc[:, ~cluster_data.columns.isin(['Customer ID'])])
  print("Creating dataframe to store the principal components...")
  principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(principal_components.shape[1])])
  principal_df['Customer ID'] = cluster_data['Customer ID']
  print("Converting dataframe to dictionary of values...")
  cluster_data_dict = principal_df.set_index('Customer ID').apply(lambda x: x.values.tolist(), axis=1).to_dict()
  print("First 3 records of dictionary...")
  first_3_records = {k: cluster_data_dict[k] for k in list(cluster_data_dict)[:3]}
  for key, value in first_3_records.items():
    print(f'{key}: {value}')
  print("Sanity check of data...")
  num_keys = len(cluster_data_dict)
  print("Number of keys:", num_keys)
  num_values = len(cluster_data_dict.values())
  print("Number of values:", num_values) 
  print("Creating check segment variable to evaluate the performance of algorithm...")
  df['check_segment'] = df['RFM'].astype(int).apply(rfm_score_to_label)
  print("Running K-Means Algorithm with Euclidean distance metric...")
  clusters, centroids = k_means(cluster_data_dict)
  print("Sanity Check...")
  values_list = clusters.get('cluster 1 CustomerID', [])
  num_values = len(values_list)
  print("Number of values:", num_values)
  print("Plotting Clusters...")
  plot_clusters_2d(cluster_data_dict, centroids, clusters)
  print("Model Evaluation...")
  evaluate_model(df, clusters, centroids, 11, principal_components, cluster_data_dict, len(list(cluster_data_dict.values())[0]), pca)


#class Centroid:
#    def __init__(self, location):
#        self.location = location
#        self.closest_users = set()

#def get_k_means(user_feature_map, num_features_per_user, k):
#    random.seed(42)
#    initial_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
    
#    centroids = {f'cluster{ik}Centroids': user_feature_map[initial_centroid_users[ik]] for ik in range(k)}
    
#    for _ in range(40):
#        clusters = {f'cluster{ik}Users': [] for ik in range(k)}
        
#        dists = {f'centroid{ik}dists': {u: sum([abs(centroids[f'cluster{ik}Centroids'][j] - user_feature_map[u][j]) for j in range(num_features_per_user)]) for u in user_feature_map} for ik in range(k)}
        
#        for u in user_feature_map:
#            tempDists = [dists[f'centroid{ik}dists'][u] for ik in range(k)]
#            clusters[f'cluster{tempDists.index(min(tempDists))}Users'].append(u)
        
#        for ik in range(k):
#            sumMean = [0] * num_features_per_user
#            N = len(clusters[f'cluster{ik}Users'])
            
#            if N != 0:
#                for u in clusters[f'cluster{ik}Users']:
#                    sumMean = [sumMean[j] + user_feature_map[u][j] / N for j in range(num_features_per_user)]
                
#                centroids[f'cluster{ik}Centroids'] = sumMean
        
#    final_clusters = {u: f'cluster{ik}Users' for ik in range(k) for u in clusters[f'cluster{ik}Users']}
#    return list(centroids.values()), dists, final_clusters

#centroids, dists, final_clusters = get_k_means(user_feature_map=cluster_data_dict, num_features_per_user=len(list(cluster_data_dict.values())[0]), k=11)

#values_list = final_clusters.get('cluster1Users', [])
#num_values = len(values_list)
#print("Number of values:", num_values)
#print(final_clusters)
#print(centroids)


#import plotly.express as px
#colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink']
#data = {'User': [], 'Principal Component 1': [], 'Principal Component 2': [], 'Cluster': [], 'Segment': []}
#for user, cluster_name in final_clusters.items():
#    user_features = cluster_data_dict[user]
    
    # Assuming 'user_id' is the common column between cluster_data_dict and df
#    user_id = user
#    if user_id in df['Customer ID'].values:
#        segment_value = df[df['Customer ID'] == user_id]['check_segment'].iloc[0]
#    else:
#        segment_value = 'Unknown'  # You can change this to a default value or handle it as needed

#    data['User'].append(user)
#    data['Principal Component 1'].append(user_features[0])  # Assuming principal_comp1 is the first element
#    data['Principal Component 2'].append(user_features[1])  # Assuming principal_comp2 is the second element
#    data['Cluster'].append(cluster_name)
#    data['Segment'].append(segment_value)

# Plot the scatter plot
#fig = px.scatter(data, x='Principal Component 1', y='Principal Component 2', color='Cluster', symbol='Cluster',
#                 title='K-Means Clustering', labels={'Principal Component 1': 'PC1', 'Principal Component 2': 'PC2'},
#                 hover_data={'User', 'Segment'}, color_discrete_map={'Cluster': colors})

# Update marker colors
#for i, color in enumerate(colors):
#    fig.for_each_trace(lambda t: t.update(marker=dict(color=color), selector=dict(name=f'Cluster {i}')))

# Show the plot
#fig.show()
