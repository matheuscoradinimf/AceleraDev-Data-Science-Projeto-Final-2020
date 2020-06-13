import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

colunas = ['id', 'fl_matriz', 'sg_uf', 'natureza_juridica_macro', 'setor',
       'idade_empresa_anos', 'fl_me', 'fl_sa', 'fl_epp', 'fl_mei', 'fl_ltda',
       'fl_st_especial', 'fl_email', 'fl_telefone', 'fl_rm', 'nm_segmento',
       'fl_spa', 'fl_antt', 'fl_veiculo', 'fl_optante_simples',
       'vl_total_veiculos_pesados_grupo', 'vl_total_veiculos_leves_grupo',
       'fl_optante_simei', 'sg_uf_matriz', 'de_saude_tributaria',
       'nu_meses_rescencia', 'de_nivel_atividade', 'fl_simples_irregular',
       'empsetorcensitariofaixarendapopulacao', 'nm_meso_regiao',
       'fl_passivel_iss', 'qt_socios', 'idade_media_socios',
       'vl_faturamento_estimado_aux', 'vl_faturamento_estimado_grupo_aux',
       'qt_filiais']

def preprocess(df, features):
    df['qt_coligados'].fillna(0, inplace=True)
    df['qt_socios'].fillna(1, inplace=True)
    df['qt_socios_pf'].fillna(1, inplace=True)
    df['qt_socios_pj'].fillna(0, inplace=True)
    df['qt_funcionarios'].fillna(0, inplace=True)
    df['tx_crescimento_12meses'].fillna(0, inplace=True)
    df['tx_crescimento_24meses'].fillna(0, inplace=True)
    df['fl_optante_simei'].fillna('False', inplace=True)
    df['fl_optante_simples'].fillna('False', inplace=True)
    df['nm_meso_regiao'].fillna('OUTROS', inplace=True)
    df['nu_meses_rescencia'].fillna(df['nu_meses_rescencia'].median(), inplace=True)
    df['vl_faturamento_estimado_aux'].fillna(df['vl_faturamento_estimado_aux'].median(), inplace=True)
    df['vl_faturamento_estimado_grupo_aux'].fillna(df['vl_faturamento_estimado_aux'].median(), inplace=True)
    df.loc[df['sg_uf_matriz'].isna(), 'sg_uf_matriz'] = df.loc[df['sg_uf_matriz'].isna(), 'sg_uf']
    df['de_nivel_atividade'].fillna('MUITO BAIXA', inplace=True)
    df['de_saude_tributaria'].fillna('VERMELHO', inplace=True)
    df['idade_media_socios'].fillna(df['idade_media_socios'].median(), inplace=True)
    df['empsetorcensitariofaixarendapopulacao'].fillna(df['empsetorcensitariofaixarendapopulacao'].median(),inplace=True)
    df['porc_st_regular'] = df['qt_socios_st_regular'] / df['qt_socios']
    df['socio_pep'] = 0
    df.loc[df['qt_socios_pep'] > 0, 'socio_pep'] = 1
    df['coligada_exterior'] = 0
    df.loc[df['qt_coligados_exterior'] > 0, 'coligada_exterior'] = 1
    df['porc_socios_pf'] = df['qt_socios_pf'] / df['qt_socios']
    df['porc_socios_pj'] = df['qt_socios_pj'] / df['qt_socios']
    df = df[df['idade_media_socios'] > 0]
    df = df[features]
    df.loc[df['fl_rm'] == 'NAO', 'fl_rm'] = 0
    df.loc[df['fl_rm'] == 'SIM', 'fl_rm'] = 1
    df.loc[:, 'fl_rm'] = pd.to_numeric(df['fl_rm'])
    col_bool = df.dtypes[df.dtypes == 'bool'].index
    for col in col_bool:
        df[col] = df[col].astype(int)
    cat_cols = df.select_dtypes('object').columns
    cat_cols = cat_cols[1:]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    scaler = MaxAbsScaler()
    df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]))
    df.fillna(0, inplace=True)
    return df

def similarity(base,portfolio, n):
    transpose = portfolio.T
    similarities = base.dot(transpose)
    similarities = similarities.mean(axis=1)
    similarities = pd.DataFrame(similarities.nlargest(n))
    similarities[0] = np.arange(n) +1
    similarities = similarities.rename(columns={0:'Rank'})
    return similarities

def SVD(df):
    svd = TruncatedSVD(65)
    svd_features = svd.fit_transform(df)
    norm_features = normalize(svd_features)
    df_svd = pd.DataFrame(norm_features)
    return df_svd

def main():
    #DATA_URL = ('https://raw.githubusercontent.com/matheuscoradini/AceleraDev-Data-Science-Projeto-Final-2020/master/Streamlit/feat1.csv')
    #data = pd.read_csv(DATA_URL)
    #st.write(data.head())


    base = st.file_uploader('Faça o upload do dataset de empresas', type='csv')
    portfolio = st.file_uploader('Faça o upload do portfólio de clientes', type='csv')
    if (base is not None) and (portfolio is not None):
        base = pd.read_csv(base)
        portfolio = pd.read_csv(portfolio)
        base_id = base['id']
        portfolio_id = portfolio['id']

        df = base.append(portfolio)
        df = preprocess(df, colunas)
        df_svd = SVD(df)

        base = df_svd[:len(base)]
        base.index = base_id
        portfolio = df_svd[len(base):]
        portfolio.index = portfolio_id

        n = st.slider('Quantas recomendações deseja?', 1, 5000, 1000)

        recomend = similarity(base, portfolio, n)

        st.write(recomend)

if __name__ == '__main__':
    main()