from pickle import FALSE
from pydoc import doc
from typing import Set
import spacy 
import numpy as np
from collections import defaultdict
from math import log
from graph import Graph
from datasets import load_dataset
from datasets import load_metric

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")

def main():
    

    example = '''    
Aug 30 (Reuters) - Mikhail Gorbachev, who ended the Cold War without bloodshed but failed to prevent the collapse of the Soviet Union, died on Tuesday at the age of 91, hospital officials in Moscow said.

Gorbachev, the last Soviet president, forged arms reduction deals with the United States and partnerships with Western powers to remove the Iron Curtain that had divided Europe since World War Two and bring about the reunification of Germany.

Advertisement · Scroll to continue
But his internal reforms helped weaken the Soviet Union to the point where it fell apart, a moment that President Vladimir Putin has called the "greatest geopolitical catastrophe" of the twentieth century.

"Mikhail Gorbachev passed away tonight after a serious and protracted disease," said Russia's Central Clinical Hospital.

Putin expressed "his deepest condolences", Kremlin spokesman Dmitry Peskov told Interfax. "Tomorrow he will send a telegram of condolences to his family and friends," he said.

Putin said in 2018 he would reverse the Soviet Union's disintegration if he could, news agencies reported.

World leaders were quick to pay tribute. European Commission chief Ursula von der Leyen said Gorbachev, who was awarded the Nobel Peace Prize in 1990, had opened the way for a free Europe.

U.S. President Joe Biden said he had believed in "glasnost and perestroika – openness and restructuring – not as mere slogans, but as the path forward for the people of the Soviet Union after so many years of isolation and deprivation."

British Prime Minister Boris Johnson, citing Putin's invasion of Ukraine, said Gorbachev's "tireless commitment to opening up Soviet society remains an example to us all".

WESTERN PARTNERSHIPS
After decades of Cold War tension and confrontation, Gorbachev brought the Soviet Union closer to the West than at any point since World War Two.

"He gave freedom to hundreds of millions of people in Russia and around it, and also half of Europe," said former Russian liberal opposition leader Grigory Yavlinsky. "Few leaders in history have had such a decisive influence on their time."

But Gorbachev saw his legacy wrecked late in life, as the invasion of Ukraine brought Western sanctions crashing down on Moscow, and politicians in both Russia and the West began to speak of a new Cold War.

"Gorbachev died in a symbolic way when his life's work, freedom, was effectively destroyed by Putin," said Andrei Kolesnikov, senior fellow at the Carnegie Endowment for International Peace.'''
    doc = nlp(example)


    sentences = [sent for sent in doc.sents]
    print(sentences)
    assert(False)
    tf_idf_vec_table = create_norm_tfidf_vec(sentences)

    xv, yv = np.meshgrid(np.arange(len(sentences)),np.arange(len(sentences)), indexing='ij')
    print(xv, yv)
    similarity_table = np.zeros(xv.shape)
    print(similarity_table)

    for i in range(len(xv)):
        for j in range(len(yv)):
            if xv[i,j] != yv[i,j]:
                similarity_table[xv[i,j], yv[i,j]] = tf_idf_vec_table[sentences[xv[i,j]]] @ tf_idf_vec_table[sentences[yv[i,j]]]
    
    # test = nx.Graph(similarity_table)
    # print('test', test)
    # pr = nx.pagerank(nx.Graph(similarity_table))
    # print(pr.values())
    # ranked = np.argsort(list(pr.values()))[::-1][:5]
    # print(ranked)
    # for rank in ranked:
    #     print(sentences[rank])

    # assert(False)
    graph = Graph(similarity_table)
    graph = generate_score(graph)
    print(sum(graph.get_scores()))
    ranked = np.argsort(graph.get_scores())[::-1][:5]
    
    #print(ranked)
    for rank in ranked:
        print(sentences[rank])
    #generate_score(graph,d)
    #print(sum(generate_score(graph, d).get_scores()))

    similarity_table = dict()    
    for sent1 in sentences:
       for sent2 in sentences: 
           similarity_table[(sent1,sent2)] = tf_idf_vec_table[sent1] @ tf_idf_vec_table[sent2]

#    print(similarity_table)

def generate_score(graph: Graph, d: float=.85):
    i = 0
    n = graph.get_num_nodes()
    
    while(i < 100):
        node_rank = dict()
        for node in graph.get_nodes():
            rank = (1-d)/n
            for neighbor in graph.get_neighbors(node):
                neighbor_sum = sum([graph.edge_weight(neighbor, out) for out in graph.get_neighbors(neighbor)])
                
                rank += d * neighbor.get_score() * graph.edge_weight(neighbor,node)/ neighbor_sum

            node_rank[node] = rank
        
        converged = 0
        for node,rank in node_rank.items():
            converged += abs(rank - node.get_score())
            node.set_score(rank)
        
        if converged <= .000001:
            return graph

        i += 1
        
    return graph





def create_norm_tfidf_vec(sentences):
    print(sentences)
    tf_idf_vec = {sent: [] for sent in sentences}
    terms = sorted(set([tok.lemma_ for sent in sentences for tok in sent if not tok.is_stop]))
    

    tf_idf = compute_doc_tf_idf(sentences)
    
    for sent in sentences:
        for term in terms:
            tf_idf_vec[sent].append(tf_idf[(term,sent)])
        tf_idf_vec[sent] /= np.linalg.norm(tf_idf_vec[sent])

    return tf_idf_vec


def compute_setence_tf(sentence):
    sentence_tf_table = defaultdict(lambda: 0)

    for tok in sentence:
        lem = tok.lemma_
        if lem in sentence_tf_table:
            sentence_tf_table[lem] += 1
        else:
            sentence_tf_table[lem] = 1

    return sentence_tf_table


def compute_tf_table(sentences):
    tf_table = dict()
    
    for sent in sentences:
        tf_table[sent] = compute_setence_tf(sent)

    return tf_table

def compute_doc_freq(terms, tf_table):
    document_freq = dict.fromkeys(terms, 0)

    for term in terms:
        for sent, sentence_tf_table in tf_table.items():
            if sentence_tf_table[term] > 0:
                document_freq[term] += 1
        print(term, document_freq[term])
    print(sum(document_freq.values()))
    return document_freq

def compute_doc_tf_idf(sentences):
    tf_table = compute_tf_table(sentences)
    terms = set([tok.lemma_ for sent in sentences for tok in sent if not tok.is_stop])
    document_freq = compute_doc_freq(terms, tf_table)

    return create_tf_idf(terms, tf_table, document_freq, len(tf_table.keys()))

    

def create_tf_idf(terms, tf_table, document_freq, N):
    tf_idf_table = defaultdict(lambda: 0)
    for term in terms:
        for sent, sentence_tf_table in tf_table.items():
            if(sentence_tf_table[term] == 0):
                tf_idf = 0
            else:
                tf_idf = sentence_tf_table[term] * (log(N / (document_freq[term]+1)) + 1)
    #        print('TERMS: ', term, 'Sentence: ', sent, 'TF:', sentence_tf_table[term], 'Doc_freq:',  document_freq[term], 'TF_IDF',tf_idf)
            tf_idf_table[(term, sent)] = tf_idf 

    return tf_idf_table, document_freq

def compute_query_tf_idf(query, doc_freq, N):
    terms = sorted(doc_freq.keys())
    tf_table = compute_tf_table(query)
    print(tf_table.keys())
    print('--------------------------------------------------------------------------')
    query_tfidf, _ = create_tf_idf(terms, tf_table, doc_freq, N)
    final_query_tfidf = dict()
    print(query_tfidf)

    query_total = sum(final_query_tfidf.values())
    for k, v in query_tfidf.items():
        final_query_tfidf[k[0]] = v/query_total
    
    
    return final_query_tfidf









    

    
def query_predict(abs_sum, i, max_len):
    pred_sum = dataset['validation'][i]['article']
    pred_sum_sentences = [sent for sent in nlp(pred_sum).sents]

    tf_idf_table, doc_freq = compute_doc_tf_idf(pred_sum_sentences)
    #print([tok.lemma_ for tok in nlp(abs_sum)])

    query_tf_idf = compute_query_tf_idf([nlp(abs_sum)], doc_freq, len(pred_sum_sentences))

    print(query_tf_idf)








dataset = load_dataset("ccdv/pubmed-summarization")
rouge = load_metric('rouge')

if __name__ == "__main__":
    #main()
    query_predict('this is a test. Sent two.', 0, 1000)
