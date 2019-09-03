import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

filepath_dict = {'tweet': 'data/train_tweets.txt'}
unlabled_tweet_path = 'data/test_tweets_unlabeled.txt'


df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['label', 'tweet'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
ut = pd.read_csv(unlabled_tweet_path, sep='delimiter', header=None)

# sentences = ['John likes ice cream', 'John hates chocolate.']
# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(sentences)
# print(vectorizer.vocabulary_)
# print(vectorizer.transform(sentences).toarray())

df_tweet = df[df['source'] == 'tweet']
tweet = df_tweet['tweet'].values
y = df_tweet['label'].values

unlabeled_tweet = np.reshape( ut.values, ( ut.values.shape[0]))

vectorizer = CountVectorizer()
vectorizer.fit(tweet)

X_train = vectorizer.transform(tweet)
X_test = vectorizer.transform(unlabeled_tweet)

clf = Pipeline([
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None,
                                      verbose=1)),
               ])
clf.fit(X_train, y.astype('int'))
y_pred = clf.predict(X_test)

f = open("predict.txt",'w')
f.write('Id Predicted\n')
for i in range(y_pred.shape[0]):
    print(y_pred[i])
    f.writelines([str(i),' ',str(y_pred[i]),'\n'])
f.close()






