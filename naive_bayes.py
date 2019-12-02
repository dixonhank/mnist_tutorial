from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

model=BernoulliNB()
model.fit(X_train,Y_train)
Y_model=model.predict(X_train)
train_accuracy=metrics.accuracy_score(Y_model,Y_train)
Y_model=model.predict(X_test)
test_accuracy=metrics.accuracy_score(Y_model,Y_test)
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))