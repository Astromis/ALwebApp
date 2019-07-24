from flask import Flask, request, session, render_template
import numpy as np
import pickle as pk

from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling
from libact.models.logistic_regression import LogisticRegression 

from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer   
import itertools
    
# configuration
DEBUG = False
# the source file with short text per line
SOURCE_FILE = 'raw_data.txt'
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

# the vocab fo mapping sentences with their context
vocab = {}
tokenized_texts = []
for line in open(SOURCE_FILE):
    sents = sent_tokenize(line, language='russian')
    tokenized_texts.append(sents)
    for s in sents:
        vocab[s] = line

tfidf = TfidfVectorizer()
# create the vectorizer and get the X
x = tfidf.fit_transform(itertools.chain(*tokenized_texts))
# form the y by randomly fill the classes.
# to not do this, the option of labeling dataset from scratch is needed
# in Dataset class
y = np.array([0, 1,0,1,0,1,0,1] + [None] * (x.shape[0] - 8))
# Create the handfull Dataset object from libact
dataset = Dataset(x, y)
# Create strategy
qs = UncertaintySampling(dataset, method='lc', model=LogisticRegression())
# create list of sentences
texts = list(itertools.chain(*tokenized_texts))
    
@app.route('/')
def show_entries():
    # Ask the sample have to be lableled 
    # and ranger the page for the first time
    ask_id = qs.make_query()
    session['ask_id'] = int(ask_id)
    return render_template('show.html', \
        text = vocab[texts[ask_id]],\
        sentence= texts[ask_id])


@app.route('/next_sampl', methods=['POST'])
def next_sampl():
    # Parse the assessor answer
    if request.form['submit_button'] == "SKIP":
        # this condition requires an option
        # to remove the sample from the dataset
        # as a garbadge
        pass
    elif request.form['submit_button'] == "0":
        dataset.update(session['ask_id'], 0)
    elif request.form['submit_button'] == "1":
        dataset.update(session['ask_id'], 1)
    session['ask_id'] = int(qs.make_query())
    return render_template('show.html', \
        text = vocab[texts[session['ask_id']]],\
        sentence= texts[session['ask_id']])

if __name__ == '__main__':

    app.run(debug=True)

