import joblib
import os
import pandas
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
        for vectorizer in vectorizers:
            model_name = f"{classifier.__class__.__name__}_{vectorizer.__class__.__name__}"
            
            # Train
            vectorize_text = vectorizer.fit_transform(train_data.v2)
            classifier.fit(vectorize_text, train_data.v1)
            
            # Score
            vectorize_text = vectorizer.transform(test_data.v2)
            score = classifier.score(vectorize_text, test_data.v1)
            print(f'{model_name}. Score: {score}')
            
            # Save model and vectorizer
            joblib.dump(classifier, f'models/{model_name}.joblib')
            joblib.dump(vectorizer, f'models/{vectorizer.__class__.__name__}.joblib')

# Load dataset
data = pandas.read_csv('spam.csv', encoding='latin-1')
learn = data[:4400]  # 4400 items for training
test = data[4400:]  # 1172 items for testing

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],
    learn,
    test
)

# Verify saved files
print("Models saved in 'models' directory.")
