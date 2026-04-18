import json
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def process_csv(data):
    def convert_clinsig(value):
        if 'Likely_benign' in value or 'Benign' in value or 'Benign/Likely_benign' in value:
            return '0'
        elif 'Likely_pathogenic' in value or 'Pathogenic' in value or 'Pathogenic/Likely_pathogenic' in value:
            return '1'
        return value
    def convert_clinsig1(value):
        if 'clinvar.P' in value:
            return '1'
        else:
            return '0'

    if 'CLNSIG' in data.columns:
        data['CLNSIG'] = data['CLNSIG'].astype(str)
        data['CLNSIG'] = data['CLNSIG'].apply(convert_clinsig1)
    else:
        print("Warning: 'CLNSIG' column not found in the data.")

    def convert(value):
        if value == '.' or isinstance(value, float):
            return None
        elif 'D' in value or 'A' in value:
            return '1'
        else:
            return '0'

    list = ['MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred','MutationTaster_pred']
    data[list] = data[list].applymap(convert)

    def convert_CADD(value):
        if value == '.':
            return None
        if float(value) > 30:
            return '1'
        else:
            return '0'

    data['CADD_pred'] = data['CADD_phred'].apply(convert_CADD)

    def convert_DANN(value):
        if value == '.':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['DANN_pred'] = data['DANN_rankscore'].apply(convert_DANN)

    def convert_VEST4(value):
        if value == '.':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['VEST4_pred'] = data['VEST4_score'].apply(convert_VEST4)

    def convert_mut(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.75:
            return '1'
        else:
            return '0'

    data['MutPred_pred'] = data['MutPred_score'].apply(convert_mut)

    def convert_mvp(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.7:
            return '1'
        else:
            return '0'

    data['MVP_pred'] = data['MVP_score'].apply(convert_mvp)

    def convert_revel(value):
        if value == '.' or value == '-':
            return None
        if float(value) > 0.5:
            return '1'
        else:
            return '0'

    data['REVEL_pred'] = data['REVEL_score'].apply(convert_revel)
    list_score = ['MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore', 'MutPred_score', 'MVP_score',
                  'CADD_phred', 'MutationTaster_score', 'VEST4_score','REVEL_score']

    def convert_score(value):
        if value == '.' or value == '-':
            return None
        else:
            return value

    data[list_score] = data[list_score].applymap(convert_score)

    return data


def reorder_columns(data):

    new_order = [
        'Chr','Start','End','Ref','Alt','Func.refGene','Gene.refGene','GeneDetail.refGene',
        'AAChange.refGene',

        'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred','DANN_pred','MutPred_pred','MVP_pred','REVEL_pred',
        'CADD_pred','MutationTaster_pred','VEST4_pred',
        'MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore','MutPred_score','MVP_score',
        'CADD_phred', 'MutationTaster_score', 'VEST4_score','REVEL_score',

        'AF', 'AF_raw','AF_male', 'AF_female', 'AF_afr', 'AF_ami', 'AF_amr',
        'AF_asj', 'AF_eas', 'AF_fin', 'AF_nfe', 'AF_oth', 'AF_sas',

        'gdi', 'gdi_phred', 'rvis1', 'rvis2', 'lof_score','func',

        'DS_AG', 'DS_AL', 'DS_DG','DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',

        'GERP++_NR','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phyloP17way_primate',
        'phastCons100way_vertebrate','phastCons30way_mammalian','phastCons17way_primate',
        'GERP++_RS_rankscore','phyloP100way_vertebrate_rankscore','phyloP20way_mammalian',
        'phyloP20way_mammalian_rankscore','phastCons100way_vertebrate_rankscore',
        'phastCons20way_mammalian','phastCons20way_mammalian_rankscore','SiPhy_29way_logOdds',
        'SiPhy_29way_logOdds_rankscore',

        'molecular_weight',
        'equipotential_point', 'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic',
        'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic',
        'basic', 'sulfur', 'pka_cooh', 'pka_nh3', 'blosum100',

        'Gm12878', 'H1hesc', 'Hepg2',
        'Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf'
    ]


    label_column = ['CLNSIG']
    print(phenotype_columns)

    new_order = new_order + phenotype_columns+ label_column

    if set(new_order) != set(data.columns):
        print("Warning: Column mismatch detected")
        missing = set(data.columns) - set(new_order)
        additional = set(new_order) - set(data.columns)
        if missing:
            print("Missing columns in new_order:", missing)
        if additional:
            print("Additional columns in new_order:", additional)

    data = data[new_order]
    return data


def replace_empty_with_zero(data):

    data.replace({'': np.nan, '.': np.nan}, inplace=True)

    num_cols = data.shape[1] - 30
    print(num_cols)

    cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'Func.refGene', 'Gene.refGene', 'GeneDetail.refGene',
           'AAChange.refGene', 'CLNSIG', 'MetaSVM_pred', 'MetaLR_pred', 'M-CAP_pred', 'DANN_pred', 'MutPred_pred',
           'MVP_pred', 'REVEL_pred',
           'CADD_pred', 'MutationTaster_pred', 'VEST4_pred',
           'MetaSVM_score', 'MetaLR_score', 'M-CAP_score', 'DANN_rankscore', 'MutPred_score', 'MVP_score',
           'CADD_phred', 'MutationTaster_score', 'VEST4_score', 'REVEL_score']
    non_chr_start_cols = [col for col in data.columns if col not in cols]
    non_null_count = data[non_chr_start_cols].apply(lambda row: sum(pd.notnull(cell) for cell in row), axis=1)
    print(non_null_count)

    non_null_rate_per_row = non_null_count / num_cols

    threshold = 0
    data = data[non_null_rate_per_row >= threshold]
    return data

input_csv = './train/merged_train.csv'
output_csv = './train/train_no_0.csv'

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
print(label_encoders['func'].classes_)
data = pd.read_csv(input_csv,low_memory=False)
phenotype_columns = list(data.columns[196:])
print(len(data))
value_counts1 = data['func'].value_counts()
print(value_counts1)

data = process_csv(data)

categorical_columns = ['func','blosum100','Gm12878', 'H1hesc', 'Hepg2','Hmec', 'Hsmm', 'Huvec', 'K562', 'Nhek', 'Nhlf',
        'hydrophilic', 'hydrophobic', 'amphipathic ', 'cyclic',
        'essential', 'aromatic', 'aliphatic', 'nonpolar', 'polar_uncharged', 'acidic','basic',
       'sulfur']

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
print(label_encoders['func'].classes_)
le = LabelEncoder()

for column in categorical_columns:
    data[column] = le.fit_transform(data[column])
data = reorder_columns(data)

X = data.iloc[:, 99:-1]
y = data['CLNSIG']

file_path = './train_hpo.json'
with open(file_path, 'r') as f:
    feature_names = json.load(f)
if (X.columns != feature_names).any():
    raise ValueError("特征名称不匹配，请检查数据框列名")
pca = joblib.load('./pca.gz')
X_pca = pca.transform(X)
data = pd.concat([data.iloc[:, :99], pd.DataFrame(X_pca),data.iloc[:,-1]], axis=1)
data = replace_empty_with_zero(data)
data.reset_index(drop=True, inplace=True)
data.columns = data.columns.astype(str)
imputer = joblib.load('./imputer.gz')
data_to_impute = data.iloc[:,29:-1]

data_to_impute.columns = data_to_impute.columns.astype(str)

data_imputed = imputer.transform(data_to_impute)
data_imputed = pd.DataFrame(data_imputed, columns=data_to_impute.columns)
data = pd.concat([data.iloc[:,:29],data_imputed, data[['CLNSIG']]], axis=1)
data.to_csv(output_csv, index=False)