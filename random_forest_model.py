from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=19,
        criterion='gini',
        max_depth=250,
        max_features=0.4,
        n_jobs=-1,
        random_state=22
    )
    model.fit(X_train, y_train)
    return model

def evaluate_rf(model, X_train, X_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f'Training accuracy: {train_acc:.4f}')
    print(f'Test accuracy:     {test_acc:.4f}')

    plt.bar(['Train', 'Test'], [train_acc, test_acc], color=['skyblue', 'salmon'])
    plt.ylim(0, 1)
    plt.title('Random Forest Accuracy')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
