import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb':   'data/imdb_labelled.txt',
                 'tweet': 'data/train_tweets.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['label', 'sentence'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])

# sentences = ['John likes ice cream', 'John hates chocolate.']
# vectorizer = CountVectorizer(min_df=0, lowercase=False)
# vectorizer.fit(sentences)
# print(vectorizer.vocabulary_)
# print(vectorizer.transform(sentences).toarray())

df_tweet = df[df['source'] == 'tweet']
sentences = df_tweet['sentence'].values
y = df_tweet['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=1000)


vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

clf = Pipeline([
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None,
                                      verbose=1)),
               ])
clf.fit(X_train, y_train.astype('int'))
y_pred = clf.predict(X_test)


print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=['may_tag']))
