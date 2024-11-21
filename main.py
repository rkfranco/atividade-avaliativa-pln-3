"""
Bibliotecas para instalar:
pip install numpy==1.24.3 sentence-transformers spacy
python -m spacy download pt_core_news_lg

Foram aplicados:
análise de tópicos
identificação de entidades
"""

import gensim
import gensim.corpora
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from spacy.tokens import Span

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('pt_core_news_lg')
# Carregar nltk
nltk.download('punkt')


def identify_entities(data: list[str], print_entities: bool = False) -> set[Span]:
    # Aplicando embedding
    documents = [nlp(str(text)) for text in data]
    entities: set[Span] = set[Span]()

    # Identificar entidades
    for doc in documents:
        for ent in doc.ents:
            entities.add(ent)
            if print_entities:
                print(f'Entidade: {ent.text}, Tipo: {ent.label_}')
    return entities


def print_topics(data: list[str]):
    # Tokenização
    texts = [word_tokenize(doc.lower()) for doc in data]

    # Criar um dicionário e um corpus
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Criar o modelo LDA
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

    # Mostrar os tópicos
    for idx, topic in lda_model.print_topics(-1):
        print(f'Tópico {idx}: {topic}')


def plot_frequency_entities(entities):
    # Criando o gráfico de colunas
    entities_count: dict[str, int] = dict()

    for ent in entities:
        if ent.text not in entities_count.keys():
            entities_count[ent.text] = 1
        else:
            entities_count[ent.text] = entities_count[ent.text] + 1

    entities_dec = dict(sorted(entities_count.items(), key=lambda item: item[1], reverse=True))
    keys = list(entities_dec.keys())[:25]

    plt.bar([key for key in keys], [entities_dec[key] for key in keys], width=0.4)

    # Adicionando título e rótulos aos eixos
    plt.title('Gráfico de Colunas')
    plt.xlabel('Entidades')
    plt.ylabel('Frequência')

    # Ajustando o tamanho da fonte dos rótulos das categorias (ticks)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Exibindo o gráfico
    plt.show()


if __name__ == '__main__':
    try:
        df = pd.read_csv('uol_news_data.csv', sep=';', encoding='utf-8-sig')
        data = df['content_without_pontuation']
        tokens_lemmatized = df['tokens_lemmatized']
        tokens_stemmed = df['tokens_stemmed']

        # Identificacao de topicos
        print(f'\nTokens com lematização')
        print_topics(tokens_lemmatized)
        print(f'\nTokens com stemização')
        print_topics(tokens_stemmed)
        print('\n')

        # Identifica entidades
        entities = identify_entities(data)
        plot_frequency_entities(entities)
    except Exception as e:
        print(f"Erro ao executar o script: {e}")
