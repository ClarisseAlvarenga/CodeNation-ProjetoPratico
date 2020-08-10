
# -*- coding: utf-8 -*-

# importando as bibliotecas mais básicas:
import streamlit as st
import pandas as pd
import base64
import sklearn
import zipfile
# import matplotlib.pyplot as plt
# import seaborn as sns
from zipfile import ZipFile
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def ajusta_arquivo_principal(mercado):
    ## CRIANDO UM DF A PARTIR DAS FEATURES SELECIONADAS DURANTE A ANÁLISE EXPLORATÓRIA
    features = ['id', 'fl_matriz',
                'de_natureza_juridica',
                'sg_uf',
                'setor',
                'idade_emp_cat',
                'fl_email',
                'fl_telefone',
                'fl_rm',
                'nm_divisao',
                'nm_segmento',
                'fl_veiculo',
                'fl_optante_simples',
                'fl_optante_simei',
                'de_saude_tributaria',
                'de_nivel_atividade',
                'nm_meso_regiao',
                'nm_micro_regiao',
                'fl_passivel_iss',
                'qt_socios',
                'idade_media_socios',
                'de_faixa_faturamento_estimado',
                'de_faixa_faturamento_estimado_grupo',
                'qt_filiais']
    dados = mercado[features].copy()
    # transformar o id em index
    dados.set_index('id', inplace=True)
    # ALTERANDO OS DADOS BOOLEANOS PARA BINÁRIOS
    dados = dados.replace({True: 1, False: 0})

    #TRANSFORMAÇÃO DE DADOS NUMÉRICOS EM CATEGÓRICOS
    # Criar classificação se contém filial ou não. Deletar "qt_filiais"
    dados['contem_filial'] = [1 if n > 0 else 0 for n in dados['qt_filiais']]
    del dados['qt_filiais']
    # transformando a coluna qt_socios em categórica
    dados.qt_socios.value_counts(normalize=True)
    pd.cut(dados.qt_socios, bins=[-1, 0, 1, 10, 50, 100, 246], include_lowest=True,
           labels=['s/info', 'único dono', 'até 10', 'entre 10 e 50', 'entre 50 e 100', 'mais de 100'])

    dados['socios_cat'] = pd.cut(dados.qt_socios, bins=[-1, 0, 1, 10, 50, 100, 246], include_lowest=True,
                                 labels=['s/info', 'único dono', 'até 10', 'entre 10 e 50', 'entre 50 e 100',
                                         'mais de 100'])

    dados.drop(['qt_socios'], axis=1, inplace=True)
    # categorizando a idade média dos sócios
    pd.cut(dados.idade_media_socios, bins=[-3, 0, 30, 50, 70, 127],
           labels=['s/info', 'até 30 anos', 'entre 30 e 50', 'entre 50 e 70', 'mais de 70 anos'], include_lowest=True)

    dados['idade_media_socios_cat'] = pd.cut(dados.idade_media_socios, bins=[-2, 0, 30, 50, 70, 127],
                                             labels=['s/info', 'até 30 anos', 'entre 30 e 50', 'entre 50 e 70',
                                                     'mais de 70 anos'])
    # excluindo a variável numérica categorizada
    dados.drop(['idade_media_socios'], axis=1, inplace=True)
    # TRATANDO NULOS
    dados.fillna('s/info', inplace=True)
    #TRANSFORMANDO TODAS AS COLUNAS EM CATEGÓRICAS
    # usando um laço para mudar o tipo da variável para string
    for col in dados.columns:
        dados[col] = dados[col].astype(str)

    return dados


def preprocessa_dados(dados):
    # instanciando o LabelEncoder
    le = LabelEncoder()
    #criando um novo arquivo para lidar com o preprocessamento
    X = dados.copy()
    # Categorical confirmando as variáveis categóricas
    categorical_feature_mask = X.dtypes == object
    # filtrar colunas categóricas e salvá-las numa lista
    categorical_cols = X.columns[categorical_feature_mask].tolist()
    # aplicando o LabelEncoder nas colunas categóricas - POR COLUNA usando uma função lambda
    X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
    return X


def cria_modelo():
    # cria o modelo
    nn = NearestNeighbors()
    return nn


def ajusta_portfolio(lista_clientes, base):
    portfolio = lista_clientes[['id']]
    portfolio = portfolio.merge(base, on='id')
    portfolio.set_index('id', inplace=True)
    return portfolio


# criando uma função de retorno
def busca_recomendacoes(portfolio, modelo, X,dados):
    #ajustando os modelos aos dados
    st.write('Ajustando o modelo aos seus dados... isso pode levar alguns minutos')
    st.image(
        'https://static.agorapulse.com/wp-content/uploads/2016/12/giphy.gif')
    st.write('Veja alguns gatinhos enquanto isso...')
    st.video('https://youtu.be/_u_bdjPsj5U')
    modelo.fit(X)
    st.write('Buscando recomendações para você...')
    st.image(
        'https://i.pinimg.com/originals/30/72/55/3072558f02be09d1a156ddfb01bd4be4.gif')
    # busca as distâncias e os índices dos vizinhos, tendo o portfolio como parâmetro
    distances, indices = modelo.kneighbors(portfolio)
    st.write('Estamos criando um arquivo para você... só mais um minutinho...')


    # cria um df legível com as variáveis originais, usando o dados como padrão
    df_cliente = pd.DataFrame([dados.iloc[indices[row,col]] for row in range(indices.shape[0]) for col in range(indices.shape[1])]).reset_index()
    return df_cliente



def main():
    st.title('PROJETO PRÁTICO CODENATION ACELERADEV DS')
    st.header('LEAD RECOMMENDER')
    st.write('Original file is located at:')
    st.write('https://colab.research.google.com/drive/1GTwzIwef1iOtInC0KBxMngjeFhZ9dSNW')
    st.subheader('**Objetivo**')
    st.write(
        'O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).')

    st.subheader('**Contextualização**')
    st.write(
        'Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.')
    st.write(
        'Além disso, sua solução deve ser agnóstica ao usuário. Qualquer usuário com uma lista de clientes que queira explorar esse mercado pode extrair valor do serviço.')

    st.subheader('**Requisitos de Negócio**')
    st.write('Para o desafio, deverão ser consideradas as seguintes bases:')
    st.write('**Mercado:** ')
    st.write(
        'Base com informações sobre as empresas do Mercado a ser considerado. Portfolio 1: Ids dos clientes da empresa 1 Portfolio 2: Ids dos clientes da empresa 2 Portfolio 3: Ids dos clientes da empresa 3')
    st.write('Obs: todas as empresas(ids) dos portfolios estão contidos no Mercado(base de população).')
    st.write('Link para download das bases Mercado, Portfolio 1, Portfolio 2 e Portfolio 3 respectivamente:')

    st.subheader('**Bases de Estudo**')
    st.write('https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_market.csv.zip')
    st.write('https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio1.csv')
    st.write('https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv')
    st.write('https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio3.csv')

    st.subheader('**Dicionário de variáveis**')
    st.write('https://s3-us-west-1.amazonaws.com/codenation-challenges/ml-leads/features_dictionary.pdf')

    st.subheader('**Análise Exploratória**')
    st.write('https://colab.research.google.com/drive/12Hi-GeEitn7gOWMCFVJl8EbvXSmaBaj0?usp=sharing')

    st.header('Vamos baixar a base de onde virão todos os portfólios:')
    st.markdown('File Uploader')
    file = st.file_uploader('Baixe o arquivo do Mercado e faça o upload em .zip', type='zip')
    st.header('Escolha um dos três portfólios e suba o arquivo csv aqui:')
    st.markdown('File Uploader')
    arquivo = st.file_uploader('Baixe o arquivo do seu portfolio em .csv', type='csv')

    # abrindo o arquivo zip usando o zipfile, abrindo todos os arquivos finalizados em csv
    if file and arquivo:
        st.subheader('**Vamos começar sua análise**')
        zip_file = ZipFile(file)
        dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
               for text_file in zip_file.infolist() if text_file.filename.endswith('.csv')}
        list(zip(dfs))

        lista_clientes = pd.DataFrame(pd.read_csv(arquivo))

        # criando o DF de mercado
        mercado = pd.DataFrame(dfs['estaticos_market.csv'])
        st.write('Estamos processando seu arquivo...')
        st.image('https://media1.tenor.com/images/aa9c780acd020eaa5b11322b869f67fa/tenor.gif?itemid=5794186')
        dados = ajusta_arquivo_principal(mercado)
        st.write('Criando um modelo de recomendação...')
        X = preprocessa_dados(dados)
        modelo = cria_modelo()

        # RESETANDO O INDEX DO X
        base = X.reset_index()
        st.header('Criando sua base de Leads')
        st.write('Processando os dados do seu portfólio...')
        st.image('https://i1.wp.com/geekiegames.geekie.com.br/blog/wp-content/uploads/2017/11/como-fazer-calculo-mental-rapidamente.gif?resize=480%2C268&ssl=1')
        portfolio = ajusta_portfolio(lista_clientes, base)

        leads = busca_recomendacoes(portfolio, modelo, X, dados)
        st.dataframe(leads.sample(50))
        st.write('Este é o tamanho de sua lista de recomendações')
        st.write(leads.shape[0])

        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = df.to_csv(encoding='utf-8')
            b64 = base64.b64encode(csv.encode())
            payload = b64.decode()
            link = "data:text/csv;base64,{payload}"
            link = link.format(payload=payload)
            href = f'<a href={link}>Download csv file</a>'
            return href

        st.subheader('Salve seu arquivo de recomendações')
        st.markdown(get_table_download_link(leads), unsafe_allow_html=True)

#if leads:
#            st.subheader('**Salve seu arquivo de leads**')
#            botao = st.button('Salvar')
#            if botao:
#                st.markdown('Clicado')
#                leads.to_csv('leads.csv')"""



if __name__ == '__main__':
    main()
