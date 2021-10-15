import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules
import networkx as nx
from collections import defaultdict
from networkx.algorithms import bipartite
import subprocess
import time

start_time = time.time()

# Loading the data
path1 = '../data/BNC_20190501_20210430_LastTouchConversion.csv'
path2 = '../data/BNC_20190501_20210430_visits.csv'
path = '../data/BNC_ConversionPaths.csv'
path3 = '../data/BNC_FormSubmit_CC_WorkTable.csv'

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
#    list_ = list_.replace('[', '["')
    list_ = list_.replace('[', ' ')
#    list_ = list_.replace(']', '"]')
    list_ = list_.replace(']', ' ')
    return list_

def appending_values(l):
    for val in l:
        val.append('Start')

def loading_and_processing_data(path1, path2):
    df1 = pd.read_csv(path1, index_col=0)
    #looking the data
    print(df1.head())
    #looking for missing values
    print(df1.info())
    print(df1.nunique())
    df2 = pd.read_csv(path2, index_col=0)
    print(df2.head())
    print(df2.info())
    print(df2.nunique())
    # Grab list of columns to iterate through
    cols1 = df1.columns
    cols2 = df2.columns
    print("cols1:", cols1)
    print("cols2", cols2)
    df1 = df1.rename(columns={"Visits": "Conversions"})
    frames = [df1, df2]
    df3 = pd.concat(frames)
    print(df3.info())
    print(df3.nunique())
    print(df3.head())
    most_visited = df3['Last Touch Channel'].value_counts().head(50)
    print(most_visited)
    sns.set(style="darkgrid")
    ax = sns.countplot(x="Last Touch Channel", data=df3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.title("Most Touched Channels")
    plt.savefig('../reports/figures/count_last_touch_channel.png')
    #plt.show()
    df4 = (df3.groupby(['Visitor_ID', 'Last Touch Channel'])['Visits'].sum().unstack().reset_index().fillna(0).set_index('Visitor_ID'))
    print(df4.info())
    print(df4.nunique)
    print(df4.head())
    return df4

def loading_and_processing_data2(path1,path2):
    df1 = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)
    df1 = df1.rename(columns={"Visits": "Conversions"})
    frames = [df1, df2]
    df3 = pd.concat(frames)
    print(df3.head(20))
    df3 = df3.sort_values(['Visitor_ID', 'Date'], ascending=[False, True])
    df3['visit_order'] = df3.groupby('Visitor_ID').cumcount() + 1
    print("THIS IS DF3 FORMATTED", df3)
    df3 = df3.fillna(0)
    print("THIS IS DF3 WITH FILLNA", df3)
    df_paths = df3.groupby('Visitor_ID')['Last Touch Channel'].aggregate(
        lambda x: x.unique().tolist()).reset_index()
    print("These are the first paths", df_paths)
    df_last_interaction = df3.drop_duplicates('Visitor_ID', keep='last')[['Visitor_ID', 'Conversions']]
    print('These are the last interactions', df_last_interaction.head())
    df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='Visitor_ID')
    print("this are the merged paths", df_paths.head())
    #df_paths['Start'] = 'Start'
    #df_paths['Null'] = 'Null'
    #df_paths['path'] = np.where(df_paths['Conversions'] == 0,
    #                            ['Start'] + df_paths['Last Touch Channel'] + ['Null'],
    #                            ['Start'] + df_paths['Last Touch Channel'] + ['Conversion'])
    #df_paths['path'] = df_paths['path'].apply(clean_alt_list)
    #df_paths['path'] = (df_paths['Start'] + df_paths['Last Touch Channel'] + df_paths['Null']).where(
    #    (df_paths['Conversions'] == 0), df_paths['Start'] + df_paths['Last Touch Channel'] + df_paths['Null'])
    print('this is the paths dataframe', df_paths.head())
    #df_paths = df_paths[['Visitor_ID', 'path']]
    #df_paths['path'] = (df_paths['Last Touch Channel']).where((df_paths['Conversions'] == 1), "Null")
    #df_paths = df_paths[['Visitor_ID', 'path']]
    #df_paths2 = df_paths[(df_paths['path'] != "Null")]

    return [df3, print("this are the paths", df_paths)]

def loading_and_processing_data3(path3):
    dfp3 = pd.read_csv(path3, index_col=0)
    print(dfp3.info())
    print(dfp3.nunique())
    print(dfp3.head())
    return dfp3, print("this is the third path", dfp3.head())

def first_model():
    data = loading_and_processing_data(path1, path2)
    channel_encoded = data.applymap(hot_encode)
    channel_conversion = channel_encoded
    frq_touch_channel = apriori(channel_conversion, min_support=0.02, use_colnames=True)
    print(frq_touch_channel)
    rules = association_rules(frq_touch_channel, metric='lift', min_threshold=0.05)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
    print(rules)
    rules.head()
    support = rules['support'].tolist()
    support = [element*100 for element in support]
    support = np.array(support)
    print(support)
    confidence=rules["confidence"].tolist()
    confidence=np.array(confidence)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(support, confidence, alpha=0.5, marker="*")
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.show()
    rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
    rules2 = rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.3)]
    rules3 = rules2.rename(columns={'antecedent support': 'antecedent_supp', 'consequent support': 'consequent_supp'})
    print(rules3)
    rules3.head(20)
    #rules3["path"] = rules3["antecedents"] + rules3["consequents"]
    #print(rules3)
    #rules4 = rules3[["VisitorID", "path", "support"]]
    #print(rules4)
    G = nx.Graph()
    G=nx.convert_matrix.from_pandas_edgelist(rules2, 'antecedents','consequents',['confidence'])
    plt.figure(figsize=(10,10))
    fig, ax=plt.subplots(figsize = (10,10))
    pos = nx.draw_shell(G, with_labels=True, ax=ax)
    plt.title("Most used linked channels leading to a conversion")
    plt.savefig('../reports/figures/connected_channels_leading_conversion.png')
    plt.show()
    frq_touch_channel = frq_touch_channel.rename(columns={"itemsets": "Channels"})
    frq_touch_channel.head()
    frq_touch_channel["Channels"] = frq_touch_channel["Channels"].apply(lambda x: list(x)[0]).astype("unicode")
    sns.set(style="darkgrid")
    ax=sns.barplot(x="Channels", y="support", data=frq_touch_channel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.title('Channels Touched Leading to Conversion')
    plt.savefig('../reports/figures/channel_leading_conversion_2.png')
    plt.show()
    return rules3


if __name__ == '__main__':
    loading_and_processing_data(path1, path2)
    loading_and_processing_data3(path3)
    first_model()
    loading_and_processing_data2(path1, path2)

print("--- %s seconds ---" % (time.time() - start_time))