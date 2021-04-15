from flask import Flask
from flask import redirect
from flask import send_file
from flask import request, redirect, url_for
from flask import render_template
import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
app = Flask(__name__)
print(app.template_folder)
@app.route("/upload_dataset", methods=['GET', 'POST'])
def render_upload_dataset():
    if request.method == 'POST':
        try:
            print(request.files)
            dataset = request.files['dataset']
            datasetName = 'dataset.csv'
            dataset.save(os.path.join('static/dataset', datasetName))
            datasetPath = 'static/dataset/{}'.format(datasetName)
            message = {'message': 'site successfully added'}
            print(message)
        except:
            message = {'message': 'error'}
            print(message)
        finally:
            return render_template("mba.html", result = '')
    return render_template("index.html")

@app.route("/mba")
def render_mba():
    try:
        select = 0
        mba('static/dataset/dataset.csv', select)
        result = {'result' : 'Market Basket Analysis Done'}
        print('mba run')
    except:
        message = {'message': 'error'}
        print(message)
    finally:
        return render_template("mba.html", result = result)
    return render_template("mba.html", result = '')

@app.route("/view_graph")
def render_view_graph():
    return render_template("view_graph.html")

@app.route("/")
def index():
    return redirect("/upload_dataset")

def mba(datasetPath, select):
    import pandas as pd
    import numpy as np
    print('inside mba function')
    print(datasetPath)
    df = pd.read_csv(datasetPath, sep=',')
    columns = list(df.columns)
    print(columns)
    items = df[columns[select]].unique()
    itemset = set(items) # ordered set
    print(itemset)
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
    freqItemsDf = apriori(encodedDf, min_support=0.2, use_colnames=True)
    rules = association_rules(freqItemsDf, metric="confidence", min_threshold=0.6)
    graph(rules)

def graph(rules):
    from mlxtend.frequent_patterns import apriori, association_rules
    import matplotlib.pyplot as plt
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.title('Support vs Confidence')
    plt.savefig('static/graph/SupportvsConfidence.png')
    plt.scatter(rules['support'], rules['lift'], alpha=0.5)
    plt.xlabel('support')
    plt.ylabel('lift')
    plt.title('Support vs Lift')
    plt.savefig('static/graph/SupportvsLift.png')
    fit = np.polyfit(rules['lift'], rules['confidence'], 1)
    fit_fn = np.poly1d(fit)
    plt.xlabel('Lift')
    plt.ylabel('Confidence')
    plt.title('Lift vs Confidence')
    plt.savefig('static/graph/LiftvsConfidence.png')





if __name__ == "__main__":
    app.run(debug=True)
