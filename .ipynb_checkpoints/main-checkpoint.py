import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

scaler = joblib.load('scaler.pkl')
df = pd.read_csv('pp_market.csv')
portfolio = pd.read_csv('estaticos_portfolio1.csv')

portfolio['qt_coligados'].fillna(0, inplace=True)
portfolio['qt_socios'].fillna(1, inplace=True)
portfolio['qt_socios_pf'].fillna(1, inplace=True)
portfolio['qt_socios_pj'].fillna(0, inplace=True)
portfolio['qt_funcionarios'].fillna(0, inplace=True)
portfolio['tx_crescimento_12meses'].fillna(0, inplace=True)
portfolio['tx_crescimento_24meses'].fillna(0, inplace=True)
portfolio['fl_optante_simei'].fillna('False',inplace=True)
portfolio['fl_optante_simples'].fillna('False',inplace=True)
portfolio['nm_meso_regiao'].fillna('OUTROS',inplace=True)
portfolio['nu_meses_rescencia'].fillna(portfolio['nu_meses_rescencia'].median(),inplace=True)
portfolio['vl_faturamento_estimado_aux'].fillna(portfolio['vl_faturamento_estimado_aux'].median(),inplace=True)
portfolio['vl_faturamento_estimado_grupo_aux'].fillna(portfolio['vl_faturamento_estimado_aux'].median(),inplace=True)
portfolio.loc[portfolio['sg_uf_matriz'].isna(), 'sg_uf_matriz'] = portfolio.loc[portfolio['sg_uf_matriz'].isna(), 'sg_uf']
portfolio['de_nivel_atividade'].fillna('MUITO BAIXA',inplace=True)
portfolio['de_saude_tributaria'].fillna('VERMELHO',inplace=True)
portfolio['idade_media_socios'].fillna(portfolio['idade_media_socios'].median(), inplace=True)
portfolio['empsetorcensitariofaixarendapopulacao'].fillna(portfolio['empsetorcensitariofaixarendapopulacao'].median(), inplace=True)

portfolio['porc_st_regular'] = portfolio['qt_socios_st_regular']/portfolio['qt_socios']
portfolio['socio_pep'] = 0
portfolio.loc[portfolio['qt_socios_pep'] > 0, 'socio_pep'] = 1
portfolio['coligada_exterior'] = 0
portfolio.loc[portfolio['qt_coligados_exterior'] > 0, 'coligada_exterior'] = 1
portfolio['porc_socios_pf'] = portfolio['qt_socios_pf']/portfolio['qt_socios']
portfolio['porc_socios_pj'] = portfolio['qt_socios_pj']/portfolio['qt_socios']

portfolio.loc[portfolio['fl_rm'] == 'NAO', 'fl_rm'] = 0
portfolio.loc[portfolio['fl_rm'] == 'SIM', 'fl_rm'] = 1
portfolio['fl_rm'] = pd.to_numeric(portfolio['fl_rm'])
col_bool = portfolio.dtypes[portfolio.dtypes == 'bool'].index
for col in col_bool:
    portfolio[col] = portfolio[col].astype(int)
cat_cols = portfolio.select_dtypes('object').columns
cat_cols = cat_cols[1:]


portfolio = pd.get_dummies(portfolio, columns=cat_cols, drop_first=True)
colunas = df.columns
portfolio = portfolio[colunas]
df = scaler.fit_transform(df)
portfolio = scaler.transform(portfolio)


svd = TruncatedSVD(65)
svd_features = svd.fit_transform(df)
norm_features = normalize(svd_features)
df_svd = pd.DataFrame(norm_features)

transpose = portfolio.T
similarities = df.dot(transpose)
similarities = similarities.mean(axis=1)
similarities.nlargest(1000)