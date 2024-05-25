# Micronet

O Projeto Micronet foi desenvolvido com o objetivo de treinar e classificar a saúde de um organismo em função do seu microbioma.

## Índice

- [Sobre](#sobre)
- [Instalação](#instalação)
- [Uso](#uso)
- [Funcionalidades](#funcionalidades)
- [Contribuindo](#contribuindo)
- [Licença](#licença)
- [Autores](#autores)
- [Agradecimentos](#agradecimentos)

## Sobre

O Projeto Micronet foi desenvolvido com o objetivo de classificar a saúde de um organismo em função do seu microbioma. Especificamente, ele foi desenvolvido utilizando como conjuntos de dados amostras de cultura de arroz saudável e doente causada pela Dickeya zeae, obtido apartir do artigo [Bez et al.](https://enviromicro-journals.onlinelibrary.wiley.com/doi/10.1111/1462-2920.15726). Este projeto utiliza vários modelos, desde os mais clássicos como SVM, MLP, Random Forest e Árvore de Decisão, até modelos mais complexos adaptado como o [Mdeep](https://github.com/lichen-lab/MDeep), que é uma rede neural convolucional, e a própria Micronet, uma rede neural totalmente conectada. Este projeto treina e classifica a saúde das amostras, com uma taxa de AUC de mais de 90%. Neste projeto, também foram utilizados os SHAP values, que explicam melhor a previsão de cada modelo. Além disso, os resultados obtidos são avaliados utilizando métricas, matriz de confusão e a geração das árvores de decisão.

## Instalação

Passos para instalar as dependências e configurar o ambiente de desenvolvimento.

```bash
# Clone o repositório
git clone https://github.com/BarryMBarque/machineLearningAndRiceMicrobiome.git

# Entre no diretório do projeto
cd machineLearningAndRiceMicrobiome

# Crie um ambiente virtual com o Conda (Versão utilizada: 4.10.3) link para instalação: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
conda create --name meu_ambiente python=3.9

# Ativar o ambiente virtaul
conda activate meu_ambiente

# Instalar as dependências
pip install -r requirements.txt

#Instalar o jupyter notebook
conda install jupyter
```

## Uso
### Utilizando os conjuntos de dados desse projeto.
```bash
#Iniciar o projeto com jupyter notebook
jupyter notebook
```
Após executar os passos de intalação, descompacte o zip datasets.zip do projeto e execute o notebook Micronet.ipynb

#### vídeo demostrativo para executar esse projeto
[Vídeo](https://drive.google.com/file/d/1TsKZFqV2b6tgyEtL5I9PBXrpTyp9fp7A/view?usp=sharing)


### Utilizando os conjuntos de dados de um outro projetos.
Lembrando que para esse projeto é consjunto de calssificadores binário, que precisam de 3 árquivos, sendo a tabela de abundancia (ASVs) das amostras classificadas como 0 ou 1, a Tabela de correlação entre as ASVs, e uma tabela que contém os Ids dos ASVs e seus repectivos nomes scientificos.

## Funcionalidades

- Previsão da saúde de organismos com base em seu microbioma.
- Análise de desempenho de diversos modelos de aprendizado de máquina.
- Visualização de métricas e interpretação de modelos através de SHAP values e árvores de decisão.

## Contribuindo

Instruções para quem deseja contribuir com o projeto.

- Faça um fork do projeto.
- Crie uma nova branch (git checkout -b feature/nova-funcionalidade).
- Faça as alterações desejadas e commit (git commit -am 'Adiciona nova funcionalidade').
- Faça o push para a branch (git push origin feature/nova-funcionalidade).
- Crie um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Autores

- Barry Malick Barque - Trabalho inicial - [Perfil](https://github.com/BarryMBarque)

## Agradecimentos

Agradeço a todos que contribuíram para o projeto de alguma forma.
Links úteis ou referências que foram utilizadas.
