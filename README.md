# AceleraDev-Data-Science-Projeto-Final-2020

http://recommender-aceleradev.herokuapp.com

## Sobre o repositório

Os notebooks estão separados por etapa, dentro de suas respectivas apstas. As etapas principais foram: EDA, pré-processamento, seleção de hiper-parâmetros, evaluating e deploy. O código do streamlit está na pasta Deploy.

## Objetivo

O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).
## Contextualização

Algumas empresas gostariam de saber quem são as demais empresas em um determinado mercado (população) que tem maior probabilidade se tornarem seus próximos clientes. Ou seja, a sua solução deve encontrar no mercado quem são os leads mais aderentes dado as características dos clientes presentes no portfólio do usuário.

Além disso, sua solução deve ser agnóstica ao usuário. Qualquer usuário com uma lista de clientes que queira explorar esse mercado pode extrair valor do serviço.

Para o desafio, deverão ser consideradas as seguintes bases:

Mercado: Base com informações sobre as empresas do Mercado a ser considerado. Portfolio 1: Ids dos clientes da empresa 1 Portfolio 2: Ids dos clientes da empresa 2 Portfolio 3: Ids dos clientes da empresa 3

## Solução

A solução final utilizada foi um sistema de recomendação do tipo Item-Based Collaborative Filter, com fatoração de matrizes e similaridade de coseno. O método utilizado para a fatoração foi o TruncatedSVD, pois obteve melhores resultados e rapidez de execução comparado ao NMF.

## Resultados

O método de avaliação consistiu na separação dos 3 portfólios em treino e teste, e utilizar as partes de treino para procurar recomendações. Quanto maior a presença dos clientes de teste nas recomendações melhor, e para isso foram utilizados os rankings da 1000, 4000, 20000 e 40000 empresas mais recomendadas.

Os 3 métodos utilizados foram a simples similaridade de coseno (Normal), a similaridade após aplicação de NMf e a similaridade após aplicação do SVD. O tempo de processamento do NMF foi de 10min 23seg, e do SVD foi de 1min 23seg.

![Alt](https://github.com/matheuscoradini/AceleraDev-Data-Science-Projeto-Final-2020/blob/master/resultados.PNG)


