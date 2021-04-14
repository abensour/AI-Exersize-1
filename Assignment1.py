import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import shap
#from catboost import CatBoostClassifier

NUM_ITERATIONS = 10
K = 5
THRESHOLD = 0.95


def impute_dataset(dataset):
    imp = IterativeImputer(max_iter=10, verbose=0)
    imp.fit(dataset)
    imputed_df = imp.transform(dataset)
    imputed_df = pd.DataFrame(imputed_df, columns=dataset.columns)
    return imputed_df


def preprocessing():
    target = 'SARS-Cov-2 exam result'
    dataset = pd.read_excel('dataset.xlsx')
    features = ['Hematocrit', 'Hemoglobin', 'Platelets', 'Red blood Cells', 'Lymphocytes',
                'Mean corpuscular hemoglobin (MCH)', 'Mean corpuscular hemoglobin concentration (MCHC)', 'Leukocytes',
                'Basophils', 'Eosinophils', 'Lactic Dehydrogenase', 'Mean corpuscular volume (MCV)',
                'Red blood cell distribution width (RDW)', 'Monocytes', 'Mean platelet volume ', 'Neutrophils',
                'Proteina C reativa mg/dL', 'Creatinine', 'Urea', 'Potassium', 'Sodium', 'Aspartate transaminase',
                'Alanine transaminase']
    clean_dataset = dataset[features]  # feature selection according to table 1
    clean_dataset = clean_dataset.loc[:, clean_dataset.isnull().sum(axis=0) < THRESHOLD * clean_dataset.shape[
        0]]  # remove features null above 95%
    len_features_without_target = clean_dataset.shape[1]  # 20 features
    clean_dataset[target] = dataset[target]
    clean_dataset = clean_dataset[
        clean_dataset.isnull().sum(axis=1) < len_features_without_target]  # remove samples that are all nan values
    target_vector = np.asarray(clean_dataset[target])
    clean_dataset.drop(target, axis='columns', inplace=True)
    imputed_dataset = impute_dataset(clean_dataset)  # complete features that are Nan
    return imputed_dataset, target_vector


def oversample(x, y):
    # transform the dataset
    oversample = SMOTE()
    return oversample.fit_resample(x, y)


def nested_cross_validation(x, y, model, space):
    total_best_model = None
    total_best_score = 0
    cv_outer = KFold(n_splits=K, shuffle=True, random_state=1)
    for train_validation_ix, test_ix in cv_outer.split(x):
        # split data
        x_train_validation, x_test = x.iloc[train_validation_ix, :], x.iloc[test_ix, :]
        y_train_validation, y_test = y[train_validation_ix], y[test_ix]
        x_train_validation, y_train_validation = oversample(x_train_validation, y_train_validation)
        cv_inner = KFold(n_splits=K, shuffle=True, random_state=1)
        search = GridSearchCV(model, space, scoring='f1', cv=cv_inner, refit=True)
        result = search.fit(x_train_validation, y_train_validation)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on test set
        predicted = best_model.predict(x_test.values)
        f1 = f1_score(y_test, predicted)
        if f1 > total_best_score:
            total_best_score = f1
            total_best_model = best_model
        print('f1=%.3f, model=%s' % (f1, result.best_params_))
    return total_best_model


def evaluate_model_using_confusion_matrix(y_test, predicted):
    # Creating a confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    conf_auroc = roc_auc_score(y_test, predicted)
    return conf_accuracy, conf_f1, conf_sensitivity, conf_specificity, conf_auroc


def evaluate_model(x, y, model):
    metrics = []
    for i in range(NUM_ITERATIONS):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        x_train, y_train = oversample(x_train, y_train)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test.values)
        accuracy, f1, sensitivity, specificity, auroc = evaluate_model_using_confusion_matrix(y_test, predicted)
        metrics.append([accuracy, f1, sensitivity, specificity, auroc])
    metrics_table = []
    metrics = np.array(metrics)
    means = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    metrics_names = ['Accuracy', 'F1-Score', 'Sensitivity', 'Specificity', 'AUROC']
    for i in range(len(metrics_names)):
        metrics_table.append(str(round(means[i], 4)) + ' +- ' + str(round(std[i], 4)))
    return pd.DataFrame(data=np.matrix(metrics_table), columns=metrics_names)


def reproduce_LR(x, y):
    # Logistic Regression
    model = LogisticRegression()
    space = dict()
    best_model_logistic = nested_cross_validation(x, y, model, space)
    evaluation_LR = evaluate_model(x, y, best_model_logistic)
    evaluation_LR.index = ['LR']
    return best_model_logistic, evaluation_LR


def reproduce_RF(x, y):
    # Random Forest
    model = RandomForestClassifier(random_state=1)
    # define search space
    space = dict()
    space['n_estimators'] = range(10, 105, 5)
    space['max_depth'] = [2, 4, 8, 16, 32, 64]
    best_model_rf = nested_cross_validation(x, y, model, space)
    evaluation_rf = evaluate_model(x, y, best_model_rf)
    evaluation_rf.index = ['RF']
    return best_model_rf, evaluation_rf


def reproduce_xgboost(x, y):
    # XGBoost
    model = xgb.XGBClassifier(n_jobs=1)
    space = dict()
    space['n_estimators'] = range(10, 105, 5)
    space['max_depth'] = [2, 4, 8, 16, 32, 64]
    space['learning_rate'] = [0.1, 0.05, 0.01]
    best_model_xgboost = nested_cross_validation(x, y, model, space)
    evaluation_xg = evaluate_model(x, y, best_model_xgboost)
    evaluation_xg.index = ['XGBoost']
    return best_model_xgboost, evaluation_xg


def reproduce_LR_RF_XGBoost(x, y):
    best_model_logistic, evaluation_LR = reproduce_LR(x, y)
    best_model_rf, evaluation_rf = reproduce_RF(x, y)
    best_model_xgboost, evaluation_xg = reproduce_xgboost(x, y)
    print(f'Evaluation:\n{pd.concat([evaluation_LR, evaluation_rf, evaluation_xg])}')
    return best_model_logistic, evaluation_LR, best_model_rf, evaluation_rf, best_model_xgboost, evaluation_xg


def add_features(x):
    new_x = x
    new_x['Hematocrit_RedBloodCells'] = np.asarray(new_x['Hematocrit'] / new_x['Red blood Cells'])
    new_x['Hemoglobin_RedBloodCells'] = np.asarray(new_x['Hemoglobin'] / new_x['Red blood Cells'])
    new_x['Leukocytes_Hematocrit'] = np.asarray(new_x['Leukocytes'] / new_x['Hematocrit'])
    new_x['Platelets_Hematocrit'] = np.asarray(new_x['Platelets'] / new_x['Hematocrit'])
    new_x['Monocytes_Leukocytes'] = np.asarray(new_x['Monocytes'] / new_x['Leukocytes'])
    return new_x


def evaluate_LGBM(x, y):
    # LGBM boost
    model = LGBMClassifier()
    space = dict()
    space['n_estimators'] = range(10, 105, 5)
    space['max_depth'] = [2, 4, 8, 16, 32, 64]
    space['learning_rate'] = [0.1, 0.05, 0.01]
    best_model_lgbm = nested_cross_validation(x, y, model, space)
    evaluation_lgbm = evaluate_model(x, y, best_model_lgbm)
    evaluation_lgbm.index = ['LGBM']
    return best_model_lgbm, evaluation_lgbm


def evaluate_CatBoost(x, y):
    # Cat Boost
    model = CatBoostClassifier()
    space = dict()
    best_model_cat = nested_cross_validation(x, y, model, space)
    evaluation_cat = evaluate_model(x, y, best_model_cat)
    evaluation_cat.index = ['CAT']
    return best_model_cat, evaluation_cat


def explain_using_shap(x, y, best_model_xgboost, best_model_cat, best_model_lgbm):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    explainer = shap.Explainer(best_model_xgboost)
    shap_values = explainer(x_train)
    shap.plots.beeswarm(shap_values)

    explainer = shap.Explainer(best_model_cat)
    shap_values = explainer(x_train)
    shap.plots.beeswarm(shap_values)

    # explain the model's predictions using SHAP
    explainer = shap.Explainer(best_model_lgbm)
    shap_values = explainer(x_train)

    # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_values[0])
    shap_values.values = shap_values.values[:, :, 1]
    shap_values.base_values = shap_values.base_values[:, 1]
    shap.plots.beeswarm(shap_values)


if __name__ == '__main__':
    # create data set
    dataset, target = preprocessing()
    x = dataset
    y = np.asarray([1 if label == 'positive' else 0 for label in target])
    print(f'The data set is\n{x}')

    print('----------- Question 2 ----------- ')
    best_model_logistic, evaluation_LR, best_model_rf, evaluation_rf, best_model_xgboost, evaluation_xg = reproduce_LR_RF_XGBoost(x, y)

    print('----------- Question 3 ----------- ')
    x = add_features(x)
    print(f'The new data set is\n{x}')
    best_model_logistic, evaluation_LR, best_model_rf, evaluation_rf, best_model_xgboost, evaluation_xg = reproduce_LR_RF_XGBoost(x,y)


    print('----------- Question 4 ----------- ')
    best_model_lgbm, evaluation_lgbm = evaluate_LGBM(x, y)
    best_model_cat, evaluation_cat = evaluate_CatBoost(x, y)
    print(f'Evaluation:\n{pd.concat([evaluation_lgbm, evaluation_cat, evaluation_xg])}')

    print('----------- Question 6 ----------- ')
    explain_using_shap(x, y, best_model_xgboost, best_model_cat, best_model_lgbm)
    