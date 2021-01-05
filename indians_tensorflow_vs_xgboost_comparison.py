import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from numpy import loadtxt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import plot_tree
import joblib

def data():
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
    x, y = dataset[:, 0:8], dataset[:, 8]
    pos = np.sum(y)
    neg = np.size(y) - pos
    initial_bias = np.log([pos / neg])
    return x, y, initial_bias

def make_k():
    return 8

def make_epoch_no():
    return 1

def k_fold_preprocessing(x, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(x, y):
        train_x, val_x = x[train_index], x[test_index]
        train_y, val_y = y[train_index], y[test_index]
        mean = train_x.mean(axis=0)
        train_x -= mean
        std = train_x.std(axis=0)
        train_x /= std
        val_x -= mean
        val_x /= std
        return train_x, train_y, val_x, val_y

def weights_reset(model, initial_bias):
    for i, layer in enumerate(model.layers):
        if hasattr(model.layers[i], 'kernel_initializer') and \
                hasattr(model.layers[i], 'bias_initializer'):
            weight_initializer = model.layers[i].kernel_initializer
            bias_initializer = model.layers[i].bias_initializer

            old_weights, old_biases = model.layers[i].get_weights()

            model.layers[i].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])
    model.layers[-1].bias.assign(initial_bias)
    return model

def create_model(x, y, initial_bias):
  input_tensor = tf.keras.Input(shape=(8,))
  dense_1 = tf.keras.layers.Dense({{choice([16, 32, 64, 128, 256])}},activation='relu')(input_tensor)
  drop_out_1 = tf.keras.layers.Dropout({{uniform(0, 0.6)}})(dense_1)
  dense_2 = tf.keras.layers.Dense({{choice([16, 32, 64, 128, 256])}}, activation='relu')(drop_out_1)
  drop_out_2 = tf.keras.layers.Dropout({{uniform(0, 0.6)}})(dense_2)
  output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(drop_out_2)
  model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
  model.layers[-1].bias.assign(initial_bias)
  epoch_no = make_epoch_no()
  k = make_k()

  val_accs = np.zeros(epoch_no)
  val_losses = np.zeros(epoch_no)

  for i in range(k):
      train_x, train_y, val_x, val_y = k_fold_preprocessing(x, y, k)
      model = weights_reset(model, initial_bias)
      history = model.fit(x=train_x,
                              y=train_y,
                              validation_data=(val_x, val_y),
                              epochs=epoch_no,
                              verbose=1,
                              batch_size=0)

      val_accs += np.array(history.history['val_accuracy'])
      val_losses += np.array(history.history['val_loss'])

  val_accs, val_losses = val_accs/k, val_losses/k
  avg_val_loss, avg_val_accuracy = np.amin(val_losses), np.amax(val_accs)

  return {'loss': -avg_val_accuracy, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          eval_space=True,
                                          trials=Trials(),
                                          functions=[make_k, make_epoch_no, k_fold_preprocessing, weights_reset])

print("\n Best performing model chosen hyper-parameters:")
print(best_run)
best_model.save('best_overnight_model.h5')
x, y, initial_bias = data()
epoch_no = make_epoch_no()
k = make_k()

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]
"""
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model_at_epoch_at_best_epoch.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
callbacks = [model_checkpoint_callback]
"""

best_model = tf.keras.models.load_model('best_overnight_model.h5')
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

val_accs = np.zeros(epoch_no)
val_losses = np.zeros(epoch_no)

metric_names = ['loss',
                'val_loss',
                'accuracy',
                'val_accuracy',
                'fp',
                'val_fp',
                'fn',
                'val_fn',
                'tp',
                'val_tp',
                'tn',
                'val_tn',
                'precision',
                'val_precision',
                'recall',
                'val_recall',
                'auc',
                'val_auc']

all_metric_results = all_metric_stdv_results = avg_metric_results = {name:[] for name in metric_names}

for i in range(k):
    train_x, train_y, val_x, val_y = k_fold_preprocessing(x, y, k)
    best_model = weights_reset(best_model, initial_bias)

    history = best_model.fit(x=train_x,
                                   y=train_y,
                                   validation_data=(val_x, val_y),
                                   epochs=epoch_no,
                                   verbose=1,
                                   batch_size=0)
    for name, value in history.history.items():
            all_metric_results[name] += [value]

all_metric_stdv_results = {name:[] for name in metric_names}
avg_metric_results = {name:[] for name in metric_names}

for name, value in all_metric_results.items():
    all_metric_stdv_results[name] = np.std(value, axis=0)
    avg_metric_results[name] = np.mean(value, axis=0)

print("\nAll metric results: ", all_metric_results)
print("\nAll average metric results", avg_metric_results)
print("\nAll average stdv metric results", all_metric_stdv_results)

loss = avg_metric_results['loss']
val_loss = avg_metric_results['val_loss']
acc = avg_metric_results['accuracy']
val_acc = avg_metric_results['val_accuracy']
false_pos = avg_metric_results['fp']
val_false_pos = avg_metric_results['val_fp']
false_neg = avg_metric_results['fn']
val_false_neg = avg_metric_results['val_fn']
precision = avg_metric_results['precision']
val_precision = avg_metric_results['val_precision']
recall = avg_metric_results['recall']
val_recall = avg_metric_results['val_recall']
AUC = avg_metric_results['auc']
val_AUC = avg_metric_results['val_auc']

epochs = range(1, epoch_no + 1)

def plot_epoch_chart(title, y_axis, dict_A, name_A, dict_B=None, name_B=None):
    all_A = all_metric_results[dict_A]
    all_B = all_metric_results[dict_B]
    mean_A = avg_metric_results[dict_A]
    mean_B = avg_metric_results[dict_B]
    upper_std_A = mean_A + all_metric_stdv_results[dict_A]
    lower_std_A = mean_A - all_metric_stdv_results[dict_A]
    upper_std_B = mean_B + all_metric_stdv_results[dict_B]
    lower_std_B = mean_B - all_metric_stdv_results[dict_B]

    plt.plot(epochs, mean_A, 'b', label=name_A)
    plt.fill_between(epochs, lower_std_A, upper_std_A, color='blue', alpha=0.2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_axis)

    if dict_B != None:
        plt.plot(epochs, mean_B, 'r', label=name_B)
        plt.fill_between(epochs, lower_std_B, upper_std_B, color='red', alpha=0.2)

    plt.legend()
    plt.show()

plot_epoch_chart(title="Training and validation loss (TF Model)",
                 y_axis="Loss",
                 dict_A="loss",
                 name_A="Training loss",
                 dict_B="val_loss",
                 name_B="Validation loss")

plot_epoch_chart(title="Training and validation accuracy (TF Model)",
                 y_axis="Accuracy",
                 dict_A="accuracy",
                 name_A="Training accuracy",
                 dict_B="val_accuracy",
                 name_B="Validation accuracy")

optimal_epochs = np.argmax(val_acc)
y_pred = []
y_actual = []

print("\nValidation accuracy was highest at: ", optimal_epochs)

for i in range(k):
    train_x, train_y, val_x, val_y = k_fold_preprocessing(x, y, k)
    model = weights_reset(best_model, initial_bias)

    history = best_model.fit(x=train_x,
                             y=train_y,
                             epochs=optimal_epochs,
                             verbose=1,
                             batch_size=0)

    k_fold_y_pred = best_model.predict(val_x)
    y_pred.extend(k_fold_y_pred)
    y_actual.extend(val_y)

y_scores = np.asarray(y_pred)
y_actual = np.asarray(y_actual)
y_pred = tf.round(y_scores)

# Now to look at results of Tensorflow model
fpr_tf, tpr_tf, thresholds_tf = roc_curve(y_actual, y_scores)
precisions_tf, recalls_tf, thresholds_tf = precision_recall_curve(y_actual, y_scores)

def plot_cm(labels, predictions, title, p=0.5):
    conf_mx = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(conf_mx, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f} '.format(p) + title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    plt.legend()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, title):
    plt.title('Precision-Recall ' + title)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Thresold')
    plt.grid(True)
    plt.legend()

print('TF Results:\n')
score = recall_score(y, y_pred)
print('recall score: {:.6f}'.format(score))
score = precision_score(y, y_pred)
print('precision score: {:.6f}'.format(score))
score = auc(fpr_tf, tpr_tf)
print('auc score: {:.6f}'.format(score))
score = auc(recalls_tf, precisions_tf)
print('precision-recall auc: {:.6f}'.format(score))

best_model = weights_reset(best_model, initial_bias)

best_model.fit(x=x,
               y=y,
               epochs=optimal_epochs,
               verbose=1)
best_model.save('best_TF_model_at_optimal_epochs.h5')

# Now train XGBoost model
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
results = cross_val_score(model, x, y, cv=kfold)
"""
print("Accuracy: %.2f%% (±%.2f%%)" % (results.mean()*100, results.std()*100))
print(results)
"""
y_pred = cross_val_predict(model, x, y, cv=kfold)
y_scores = cross_val_predict(model, x, y, cv=kfold, method='predict_proba')
y_scores = y_scores[:, 1]
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y, y_scores)
precisions_xg, recalls_xg, thresholds_xg = precision_recall_curve(y, y_scores)
joblib.dump(model, 'xgb_model')

#Compare TF and XGB results
plt.plot(fpr_tf, tpr_tf, linewidth=2, label='TF')
plot_roc_curve(fpr_xg, tpr_xg, 'XGBoost')
plt.show()

plot_precision_recall_vs_threshold(precisions_tf, recalls_tf, thresholds_tf, '(TF model)')
plt.show()
plot_precision_recall_vs_threshold(precisions_xg, recalls_xg, thresholds_xg, '(XGBoost model)')
plt.show()

plot_cm(y_actual, y_pred, '(TF Model)')
plt.show()
plot_cm(y, y_pred, '(XGBoost)')
plt.show()

print('XGBoost Results:\n')
score = recall_score(y, y_pred)
print('recall score: {:.6f}'.format(score))
score = precision_score(y, y_pred)
print('precision score: {:.6f}'.format(score))
score = auc(fpr_xg, tpr_xg)
print('auc score: {:.6f}'.format(score))
score = auc(recalls_xg, precisions_xg)
print('precision-recall auc: {:.6f}'.format(score))

model.fit(x,y)
plot_tree(model)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('XGBoost tree.png')
