import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
import scikitplot as skplt
import textwrap
from umap import UMAP
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

import datapane as dp

def customwrap(s,width=30):
    return "<br>".join(textwrap.wrap(s,width=width))

def umap_plot(X_train, y_train, X_train_df):
    umap = UMAP(metric='cosine')
    umap.fit(X_train_df)
    data = pd.DataFrame(
    data = umap.transform(X_train_df),
    index = X_train_df.index,
    columns=["x","y"]
    ).assign(text=X_train).assign(color=y_train.astype(str))

    #data['text'] = data['text'].apply(lambda x:customwrap(x,90))
    
    fig = px.scatter(data,x='x',y='y',color='color',
                    color_discrete_sequence=['blue','orange'], opacity=0.6,
                    hover_data={'x':False,
                                'y':False,
                                'Review':data['text']})
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0))
    fig.update_layout(hoverlabel={'font_size':12,'bgcolor':'black'})
    fig.update_traces(marker={'size':4})
    return fig

def tt_split():
    train = pd.read_pickle('train.pkl')
    train = train.sample(frac=1).reset_index(drop=True)
    test = pd.read_pickle('test.pkl')
    test = test.sample(frac=1).reset_index(drop=True)
    X_train = train['review']
    y_train = train['score']
    X_test = test['review']
    y_test = test['score']
    return X_train,X_test,y_train,y_test

def vectorize_sentence(sentence,model):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    a = []
    for i in tokenizer(sentence):
        try:
            a.append(model.get_vector(str(i)))
        except:
            pass
        
    a=np.array(a).mean(axis=0)
    a = np.zeros(300) if np.all(a!=a) else a
    return a

def plot_feature_importance(cols,importance,ax,n=10, n_=0,importance_name='Importance'):
    feat_importance = pd.DataFrame(zip(cols, importance)
             ,columns=['Feature',importance_name]).sort_values(by=importance_name, ascending=False).iloc[n_:n,:]

    sns.barplot(y="Feature", x=importance_name, data=feat_importance,ax=ax,palette='flare')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=40, ha='right')
    return plt

def model_summary_plot(model,X_test,y_test, cols, plot_title, importance, n=10):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    fig, ax = plt.subplots(ncols=3,nrows=2, figsize=(22,10))
    
    plot_roc_curve(model,X_test,y_test,ax=ax[0,0])
    
    plot_precision_recall_curve(model,X_test,y_test,ax=ax[0,1])
      
    plot_feature_importance(cols,importance,n=10, ax=ax[1,0])
    ax[1,0].set_title('Top Feature Importances')
    
    sns.histplot(y_prob[:,1], kde=False, color=sns.color_palette()[1],ax=ax[1,1])
    ax[1,1].set_title('Predicted Probabilities')
    
    skplt.metrics.plot_confusion_matrix(y_test, model.predict(X_test), ax=ax[1,2])
    
    plt.suptitle(f'{plot_title} - Model Accuracy: {round(model.score(X_test,y_test),3)}',fontsize=20)
    
    fig.delaxes(ax[0,2])
    
    return fig

def dp_publish(fig, name):
    report = dp.Report(dp.Plot(fig) ) 
    report.publish(name=name, open=False, visibility='PUBLIC')