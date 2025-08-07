from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from data_loader import load_dataset, encode_labels
from random_forest_model import train_rf, evaluate_rf
from neural_net_model import build_nn, train_nn, plot_history, summarize_metrics

# Path to the CSV file
csv_path = "Disease and symptoms dataset.csv"

# Load data
x, y, diseases = load_dataset(csv_path)
y_encoded = encode_labels(y, diseases)

# Train/test split
X_train, X_test, y_train_rf, y_test_rf = train_test_split(x, y_encoded, test_size=0.2, random_state=22)

# Random Forest
rf_model = train_rf(X_train, y_train_rf)
evaluate_rf(rf_model, X_train, X_test, y_train_rf, y_test_rf)

# Neural Network
num_classes = len(diseases)
y_train_nn = to_categorical(y_train_rf, num_classes)
y_test_nn = to_categorical(y_test_rf, num_classes)

model = build_nn(x.shape[1], num_classes)
history = train_nn(model, X_train, y_train_nn, X_test, y_test_nn)
plot_history(history)
summarize_metrics(history)
