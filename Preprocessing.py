import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_dataset(dataset):
    imp = IterativeImputer(max_iter=10, verbose=0)
    imp.fit(dataset)
    imputed_df = imp.transform(dataset)
    imputed_df = pd.DataFrame(imputed_df, columns=dataset.columns)
    return imputed_df


def preprocessing():
    dataset = pd.read_excel('/dataset.xlsx')
    features = ['Hematocrit', 'Hemoglobin', 'Platelets', 'Red blood Cells', 'Lymphocytes',
                'Mean corpuscular hemoglobin concentration (MCHC)', 'Mean corpuscular hemoglobin (MCH)', 'Leukocytes',
                'Basophils', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Red blood cell distribution width (RDW)',
                'Monocytes', 'Mean platelet volume ', 'Neutrophils', 'Proteina C reativa mg/dL', 'Creatinine', 'Urea',
                'Potassium', 'Sodium', 'Aspartate transaminase', 'Alanine transaminase']
    features_with_label = features.copy()
    target = 'SARS-Cov-2 exam result'
    features_with_label.append(target)
    dataset = dataset[features_with_label]  # feature selection according to table 1
    dataset = dataset[dataset.isnull().sum(axis=1) < len(features)]  # remove samples that are all nan values
    imputed_dataset = impute_dataset(dataset[features])  # complete features that are Nan
    target_vector = pd.DataFrame(data=dataset[target])
    print(imputed_dataset)
    print(target_vector)
    # return impute_dataset, target_vector


preprocessing()
