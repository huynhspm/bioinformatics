import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib 

N_CORES = 5

ARGS = {
    'max_width' : 100,
    'train_split' : 0.95,
    'shuffle_size' : 1000,
    'batch_size' : 512,
}
input_dim = int(6)  # A, C, G, T, N, M
n_positions = ARGS['max_width'] * 2

# datapath
kaggle_dataset_path = 's3.msi.umn.edu/gongx030/projects/dream_PGE/training_data/pct_ds=1'

# Load the data
ds = tf.data.experimental.load(kaggle_dataset_path)

n = int(ds.cardinality())
n_train = int(n * ARGS['train_split'])
print('downsampled dataset size: %d' % (n))
print('training dataset size: %d' % (n_train))

# Shuffle the dataset
ds = ds.shuffle(ARGS['shuffle_size'], seed=1)

# Split into train and test
train_ds = ds.take(n_train)
test_ds = ds.skip(n_train)

# Reduce both datasets to 1/20th of their original sizes
train_cardinality = int(train_ds.cardinality())
test_cardinality = int(test_ds.cardinality())
print('# training samples (original): %d' % (train_cardinality))
print('# test samples (original): %d' % (test_cardinality))

datapoint = next(iter(train_ds))

for key in datapoint.keys():
    value = datapoint[key]  # Truy cập giá trị theo key
    print(f"Key: {key}")
    print(f"Shape: {value.shape}")
    print(f"Value: {value.numpy() if isinstance(value, tf.Tensor) else value}")
    print("-" * 50)


def tf_dataset_to_numpy(dataset, length=-1):
    inputs = []
    labels = []
    for sample in dataset:
        inputs.append(sample['seq'].numpy())
        labels.append(sample['expression'].numpy())
        length -= 1
        if length == 0:
            break
    return np.array(inputs), np.array(labels)

X_train, y_train = tf_dataset_to_numpy(train_ds, length=-1)
X_test, y_test = tf_dataset_to_numpy(test_ds, length=-1)

print("Inputs shape:", X_train.shape, X_test.shape)
print("Labels shape:", y_train.shape, y_test.shape)

models = {
#    "RF": RandomForestRegressor(
#        min_samples_leaf=5, random_state=0, n_jobs=N_CORES
#    ),
    "LR": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

param_grids = {
#    "RF": {"n_estimators": [5, 20, 50, 100]},
    "LR": {"positive": [True, False]},
    "KNN": {"n_neighbors": [3, 5, 20, 50]},
}

def pearson_r(x, y):
    mx = np.mean(x, axis=0, keepdims=True)
    my = np.mean(y, axis=0, keepdims=True)
    xm = x - mx
    ym = y - my

    t1_norm = xm / np.linalg.norm(xm, axis=0, keepdims=True)
    t2_norm = ym / np.linalg.norm(ym, axis=0, keepdims=True)

    return np.sum(t1_norm * t2_norm)

def test_score(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
    pearson = pearson_r(y_test, y_test_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    return r2, pearson, mse

for model_name, model in models.items():
    print(f"Training model: {model_name}")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring="r2",
        cv=5,
        n_jobs=N_CORES
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{model_name}_best_model.pkl")
    print(f"Model {model_name} saved as: {model_name}_best_model.pkl")
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best R2 score (CV) for {model_name}: {grid_search.best_score_:.4f}")
    
    r2, pearson, mse = test_score(best_model, X_test, y_test)
    print(f"Test results for {model_name}:")
    print(f"  - R2 Score: {r2:.4f}")
    print(f"  - Pearson Correlation: {pearson:.4f}")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}\n")
