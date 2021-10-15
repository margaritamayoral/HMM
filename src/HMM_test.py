## Test dataset creation
import torch
import pyro

#Hidden Markov Model
#Conversion path study
#The objective in the first analysis is to find from the tables:

#BNC_20190501_20210430_Conversion.csv
#BNC_20190501_20210430_visits.csv
#The path that the user follows and that lead to conversion in a determined time or step number.

#References
#HMM
#https://www.blopig.com/blog/2021/05/hidden-markov-models-in-python-a-simple-hidden-markov-model-with-known-emission-matrix-fitted-with-hmmlearn/

#https://web.stanford.edu/~jurafsky/slp3/

#https://web.stanford.edu/~jurafsky/slp3/8.pdf

#https://www.linkedin.com/pulse/marketing-attribution-challenge-hidden-markov-model-sudipt-roy-phd/

#https://www.krannert.purdue.edu/academics/mis/workshop/papers/Attribution_Abhishek_Fader_Hosanagar.pdf

# Defining the hot encoding function t make the data suitable
# for the concerned libraries
def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

path = "/content/drive/My Drive/Data/BNC/Credit_card.csv"
path1 = "/content/drive/My Drive/Data/BNC/BNC_20190501_20210430_Conversion.csv"
path2 = "/content/drive/My Drive/Data/BNC/BNC_20190501_20210430_visits.csv"
path3 = "/content/drive/My Drive/Data/BNC/BNC_ConversionPaths.csv"
path4 = "/content/drive/My Drive/Data/BNC/BNC_FormSubmit_CC_WorkTable.csv"

df = pd.read_csv(path, index_col=0)
df1 = pd.read_csv(path1, index_col=0)
df2 = pd.read_csv(path2, index_col=0)
dfp3 = pd.read_csv(path3, index_col=0)
dfp4 = pd.read_csv(path4, index_col=0)

#looking for missings, kind of data and shape:
print(df1.info())
#looking for unique values
print(df1.nunique())
#Looking the data
print(df1.head())

df1 = df1.rename(columns={"Visits" : "Conversions"})

#looking for missings, kind of data and shape:
print(df1.info())
#looking for unique values
print(df1.nunique())
#Looking the data
print(df1.head())

#looking for missings, kind of data and shape:
print(df2.info())
#looking for unique values
print(df2.nunique())
#Looking the data
print(df2.head())

frames = [df1, df2]
df3 = pd.concat(frames)
#looking for missings, kind of data and shape:
print(df3.info())
#looking for unique values
print(df3.nunique())
#Looking the data
print(df3.head())
df3.head()

#looking for missings, kind of data and shape:
print(dfp4.info())
#looking for unique values
print(dfp4.nunique())
#Looking the data
print(dfp4.head())
dfp4.head()

most_visited = df3['Last Touch Channel'].value_counts().head(50)
print(most_visited)

sns.set(style="darkgrid")
ax = sns.countplot(x="Last Touch Channel", data=df3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title('Most Touched Channels')
plt.savefig('/content/drive/My Drive/Data/BNC/output_path_conversions/count_last_touch_channel.png')
plt.show()

df4 = (df3.groupby(['Visitor_ID', 'Last Touch Channel'])['Visits'].sum().unstack().reset_index().fillna(0).set_index('Visitor_ID'))

#looking for missings, kind of data and shape:
print(df4.info())
#looking for unique values
print(df4.nunique())
#Looking the data
print(df4.head())
df4.head()

channel_encoded = df4.applymap(hot_encode)
channel_conversion = channel_encoded
frq_touch_channel = apriori(channel_conversion, min_support = 0.02, use_colnames = True)
print(frq_touch_channel)
#frq_touch_channel = frq_touch_channel.rename(columns={"itemsets" : "Channels"})
#frq_touch_channel.head()
rules = association_rules(frq_touch_channel, metric = 'lift', min_threshold = 0.05)
rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
print(rules)
rules.head()
support=rules['support'].tolist()
support=[element*100 for element in support]
support=np.array(support)
print(support)
confidence=rules['confidence'].tolist()
confidence=np.array(confidence)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(support, confidence,   alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()
rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])
#print(len(rules))
rules2 = rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.3)]
rules3 = rules2.rename(columns={'antecedent support': 'antecedent_supp', 'consequent support': 'consequent_supp'})
print(rules3)
rules3.head()
G = nx.Graph()
G=nx.convert_matrix.from_pandas_edgelist(rules2, 'antecedents', 'consequents', ['confidence'])
plt.figure(figsize=(10,10))
#plt.subplot(1,2,1)
fig, ax = plt.subplots(figsize = (10,10))
pos = nx.draw_shell(G, with_labels=True, ax=ax)
plt.title("Most used linked channels leading to a conversion")
plt.savefig('/content/drive/My Drive/Data/BNC/output_path_conversions/connected_channels_leading_conversion.png')
plt.show()
frq_touch_channel = frq_touch_channel.rename(columns={"itemsets" : "Channels"})
frq_touch_channel.head()
frq_touch_channel["Channels"] = frq_touch_channel["Channels"].apply(lambda x: list(x)[0]).astype("unicode")
sns.set(style="darkgrid")
ax = sns.barplot(x="Channels", y="support", data=frq_touch_channel)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title('Channels Touched Leading to Conversion')
plt.savefig('/content/drive/My Drive/Data/BNC/output_path_conversions/channel_leading_conversion_2.png')
plt.show()
df_MC_2 = df3.fillna(1)
dfp3_MC_21 = dfp3['Visits'].fillna(1)
df_MC_3 = df_MC_2.groupby(['Visitor_ID', 'Date', 'Last Touch Channel'])['Visits'].sum().to_frame()
df_MC_6 = df_MC_3['Visits'].groupby('Visitor_ID', group_keys=False)
df_MC_7 = df_MC_6.apply(lambda x: x.sort_values(ascending=False).head(3))
df_MC_8 = df_MC_3['Visits'].groupby('Visitor_ID', group_keys=False).nlargest(3)df
df_MC_9 = df_MC_2.sort_values(['Date'], ascending=True).groupby(['Visitor_ID', 'Date', 'Last Touch Channel'])['Visits'].sum().to_frame()



transition = torch.Tensor([[0.9, 0.05, 0.05],
                           [0.05, 0.9, 0.05],
                           [0.05, 0.05, 0.9],
                           ]
                          )
emission = torch.Tensor(
    [
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.3, 0.5],
    ])

samples = []

for j in torch.range(0, 50):
    observations = []

    state = pyro.sample("x_{}_0".format(j),
                        dist.Categorical(torch.Tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])),
                        )
    emission_p_t = emission[state]
    observation = pyro.sample("y_{}_0".format(j),
                              dist.Categorical(emission_p_t),
                              )
    for k in torch.range(1, 50):
        transition_p_t = transition[state]

        state = pyro.sample("x_{}_{}".format(j, k),
                            dist.Categorical(transition_p_t),
                            )
        emission_p_t = emission[state]

        observation = pyro.sample("y_{}_{}".format(j, k),
                                  dist.Categorical(emission_p_t),
                                  )

        observations.append(observation)

    sample = torch.Tensor(observations)
    samples.append(sample)
sequences = torch.vstack(samples)
lengths = torch.Tensor(np.array([len(sequence) for sequence in sequences_tensor])).to(torch.int)