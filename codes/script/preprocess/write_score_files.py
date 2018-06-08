import sys
sys.path.append('./')
from codes.func.methods import *
import os

if not os.path.exists('models/wea'):
    os.mkdir('models/wea')

if not os.path.exists('models/fv+pca'):
    os.mkdir('models/fv+pca')

if not os.path.exists('models/fv+cca'):
    os.mkdir('models/fv+cca')

wea = WEA()
wea.save_score_file('val', 'models/wea/res_val.csv')
wea.save_score_file('test', 'models/wea/res_test.csv')

fv_pca = FV_PCA()
fv_pca.save_score_file('val', 'models/fv+pca/res_val.csv')
fv_pca.save_score_file('test', 'models/fv+pca/res_test.csv')

fv_cca = FV_PCA()
fv_cca.save_score_file('val', 'models/fv+cca/res_val.csv')
fv_cca.save_score_file('test', 'models/fv+cca/res_test.csv')

if not os.path.exists('models/Ens_avr'):
    os.mkdir('models/Ens_avr')
    
if not os.path.exists('models/Ens_fv+pca'):
    os.mkdir('models/Ens_fv+pca')

if not os.path.exists('models/Ens_fv+cca'):
    os.mkdir('models/Ens_fv+cca')

for split in ['val', 'test']:
    df = get_model_ensemble_df('models/avr-None/res_%s.csv'%split, 'models/avr-vgg/res_%s.csv'%split)
    df.to_csv('models/Ens_avr/res_%s.csv'%split)
    
    df = get_model_ensemble_df('models/fv+pca-None/res_%s.csv'%split, 'models/fv+pca-vgg/res_%s.csv'%split)
    df.to_csv('models/Ens_fv+pca/res_%s.csv'%split)

    df = get_model_ensemble_df('models/fv+cca-None/res_%s.csv'%split, 'models/fv+cca-vgg/res_%s.csv'%split)
    df.to_csv('models/Ens_fv+cca/res_%s.csv'%split)