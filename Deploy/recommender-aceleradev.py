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

@st.cache
def SVD(df):
    svd = TruncatedSVD(65)
    svd_features = svd.fit_transform(df)
    norm_features = normalize(svd_features)
    df_svd = pd.DataFrame(norm_features)
    return df_svd

def main():
    st.title('Recomendador de Leads AceleraDev')
    st.image('filtro.png',use_column_width=True)
    st.markdown('### Sobre')
    st.markdown('Demo do projeto final da aceleração de Data Science da Codenation. Assim que os uploads forem feitos '
                'a recomendação estará disponível logo abaixo. Os leads são representados por seu id e seu ranking '
                'de recomendação')
    st.markdown('O limite máximo de arquivo csv aceito pela plataforma é 100mb, mas se for preciso algo maior que'
                ' isso é possível fazer o upload em partes')
    st.sidebar.markdown('by: Matheus Coradini')
    st.sidebar.markdown('GitHub: https://github.com/matheuscoradini')
    st.sidebar.markdown('Linkedin: https://www.linkedin.com/in/matheus-coradini/')
    st.sidebar.markdown('email: coradinimatheus1@gmail.com')
    radio= st.radio('Mais de um arquivo csv para o dataset base?', ['Não','Sim'])
    if radio == 'Não':
        base = st.file_uploader('Faça o upload csv do dataset de empresas', type='csv')
        if base is not None:
            base = pd.read_csv(base, index_col='Unnamed: 0')
    else:
        quantos = st.slider('Em quantos arquivos o dataset foi dividido?',2, 5)
        base = pd.DataFrame()
        csvs = []
        for i in range(quantos):
            i = st.file_uploader(f'Faça o {i+1}º upload csv do dataset de empresas', type='csv')
            csvs.append(i)
        if None not in csvs:
            for k in csvs:
                base = base.append(pd.read_csv(k, index_col='Unnamed: 0'))

    portfolio = st.file_uploader('Faça o upload csv do portfólio de clientes', type='csv')
    if (base is not None) and (portfolio is not None):
        portfolio = pd.read_csv(portfolio, index_col='Unnamed: 0')
        try:
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
        except:
            st.markdown("### Faça novamente o upload dos csvs e aguarde")
if __name__ == '__main__':
    main()