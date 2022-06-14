import pandas as pd
import matplotlib.pyplot as plt



class BaseIndex:

    def _get_link_and_anchor_text(self, line, separator):
        line_list = line.split(separator)
        link = line_list[0]
        anchor_text = None
        
        if (len(line_list) == 2):
            anchor_text = line_list[1]
        elif (len(line_list) > 2):
            print('get_link_and_anchor_text() invalid row')
            
        return (link, anchor_text)


    def _plot_most_common_lemmas(self, lemmas, counts):
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.xlabel('Lemma')
        plt.ylabel('Count') 
        plt.bar(lemmas, counts)
        plt.show()


    def calc_overall_statistics(self, input_file_path):
        stats_dict = {
            'unique_lemma_dict': {},
            'unique_non_lemma_dict': {},
            'unique_lemma_count': 0,
            'unique_non_lemma_count': 0,
            'unique_word_count': 0
        }
        
        with open(input_file_path, 'r', encoding='UTF-8') as input_file:
            for line in input_file:
                lemma, non_lemma = self._get_link_and_anchor_text(line, '|')
                lemma = lemma.strip('\n')

                if (lemma not in stats_dict['unique_lemma_dict'] and lemma not in stats_dict['unique_non_lemma_dict']):
                    stats_dict['unique_lemma_dict'][lemma] = 1
                    stats_dict['unique_word_count'] += 1
                    stats_dict['unique_lemma_count'] += 1
                elif (lemma in stats_dict['unique_lemma_dict']):
                    stats_dict['unique_lemma_dict'][lemma] += 1

                if (non_lemma):
                    non_lemma = non_lemma.strip('\n')
                    if (non_lemma not in stats_dict['unique_non_lemma_dict']):
                        stats_dict['unique_non_lemma_dict'][non_lemma] = 1
                        stats_dict['unique_word_count'] += 1
                        stats_dict['unique_non_lemma_count'] += 1
                    else:
                        stats_dict['unique_non_lemma_dict'][non_lemma] += 1
                    
        print(f"Number of unique words: {stats_dict['unique_word_count']}")
        print(f"Number of unique lemmas: {stats_dict['unique_lemma_count']}")
        print(f"Number of unique non-lemmas: {stats_dict['unique_non_lemma_count']}")

        stats_df = pd.DataFrame(stats_dict['unique_lemma_dict'].items(), columns=['lemma', 'count'])
        top_10_lemma_df = stats_df.sort_values(by='count', ascending=False)[['lemma', 'count']].head(10)
        self._plot_most_common_lemmas(top_10_lemma_df['lemma'], top_10_lemma_df['count'])



class IndexLemma(BaseIndex):

    def __init__(self):
        self.index_dict = {}
    
    
    def create_index(self, input_file_path):
        with open(input_file_path, 'r', encoding='UTF-8') as input_file:
            for line in input_file:
                lemma, non_lemma = self._get_link_and_anchor_text(line, '|')
                lemma = lemma.strip('\n')
                
                if (lemma not in self.index_dict):
                    self.index_dict[lemma] = set()
                
                if (non_lemma):
                    non_lemma = non_lemma.strip('\n')
                    self.index_dict[lemma].add(non_lemma)
    
    
    def lookup_query(self, query):
        term_list = query.split(' ')
        term_list = [term.lower() for term in term_list]

        for term in term_list:
            if term in self.index_dict:
                print(f'Lemma: {term}, non-lemma: {self.index_dict[term]}')



class IndexNonLemma(BaseIndex):

    def __init__(self):
        self.index_dict = {}
    
    
    def create_index(self, input_file_path):
        with open(input_file_path, 'r', encoding='UTF-8') as input_file:
            for line in input_file:
                lemma, non_lemma = self._get_link_and_anchor_text(line, '|')
                lemma = lemma.strip('\n')
                
                if (non_lemma):
                    non_lemma = non_lemma.strip('\n')
                    if (non_lemma not in self.index_dict):
                        self.index_dict[non_lemma] = lemma
    
    
    def lookup_query(self, query):
        term_list = query.split(' ')
        term_list = [term.lower() for term in term_list]

        for term in term_list:
            if term in self.index_dict:
                print(f'Non-lemma: {term}, lemma: {self.index_dict[term]}')
