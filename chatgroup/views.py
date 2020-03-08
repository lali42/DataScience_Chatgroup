from django.shortcuts import render
from joblib import load
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump

data = fetch_20newsgroups()
categories = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.motorcycles','sci.electronics']
train = fetch_20newsgroups(subset='train', categories=categories)

# Create your views here.
def index(req):
    train = fetch_20newsgroups(subset='train', categories=categories)
    test = fetch_20newsgroups(subset='test', categories=categories)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train.data, train.target)
    labels = model.predict(test.data)
    test.target[0:10]
    n = len(test.data)
    corrects = [ 1 for i in range(n) if test.target[i] == labels[i] ]
    corrects1 = round(sum(corrects)*100/n,2)

    dump(model, 'chatgroup.model')
    model = load('./chatgroup/static/chatgroup.model')
    label = ""
    chat  = ""
    if req.method == 'POST':
        print("POST IN")
        chat = str(req.POST['chat'])
        pred = model.predict([chat])
        label = train.target_names[pred[0]]
    return render(req, 'chatgroup/index.html' ,{
            'label':label,
            'chat':chat,
            'corrects1':corrects1,
    })