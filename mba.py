''' make a dataframe of the retail dataset '''
df = pd.read_csv('/content/retail_dataset.csv', sep=',')
df.head()

''' finds out how many unique items are actually there in the table '''
items = (df['0'].unique())
items

''' apriori module requires a dataframe that has either 0 and 1 or True and
    False as data so we have to pre-process our data'''
itemset = set(items) # ordered set
encodedVals = [] # list to store the binary encoded values
for index, row in df.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uncommon in uncommons:
        labels[uncommon] = 0
    for common in commons:
        labels[common] = 1
    encodedVals.append(labels)
encodedVals[0]
encodedDf = pd.DataFrame(encodedVals) # encoded dataframe

''' min_support can be set between 0 to 1 is a parameter supplied to the Apriori
    algorithm in order to prune candidate rules by specifying a minimum
    lower bound for the Support measure of resulting association rules '''
freqItemsDf = apriori(encodedDf, min_support=0.2, use_colnames=True)
freqItemsDf # shows the first 9 rows

''' Frequent if-then associations called association rules
    which consists of an antecedent (if) and a consequent (then).
    Metrics can be set to confidence, lift, support, leverage and conviction.
'''
rules = association_rules(freqItemsDf, metric="confidence",
                                                       min_threshold=0.6)
rules.head()# shows first 5 rows

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'],
fit_fn(rules['lift']))
plt.xlabel('Lift')
plt.ylabel('Confidence')
plt.title('Lift vs Confidence')
plt.show()
