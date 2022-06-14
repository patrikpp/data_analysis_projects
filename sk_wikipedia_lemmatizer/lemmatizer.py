import re
import html
import numpy as np
import pyspark
from pyspark.sql import SparkSession



class BaseWikiLemmatizer:

    def print_first_n_lines_of_file(self, file_path, n):
        count = 0
        
        with open(file_path, 'r', encoding='UTF-8') as input_file:
            for line in input_file: 
                print(line)
                count += 1
                
                if (count >= n):
                    break


    def _check_if_string_contains_any_letters(self, string):
        if (string and re.search('[a-zA-Z]', string)): 
            return True
        return False


    def _get_link_and_anchor_text(self, line, separator):
        line_list = line.split(separator)
        link = line_list[0]
        anchor_text = None
        
        if (len(line_list) == 2):
            anchor_text = line_list[1]
        elif (len(line_list) > 2):
            print('get_link_and_anchor_text() invalid row')
            
        return (link, anchor_text)


    def _is_stop_word(self, string, stop_words):    
        if (len(string) <= 2):
            return True
        elif (string in stop_words): 
            return True
        return False


    def _clean_tokenized_string_list(self, string_list, stop_words):
        cleaned_string_list = []
        
        for string in string_list:
            string = string.lower()
            if (self._check_if_string_contains_any_letters(string) and not self._is_stop_word(string, stop_words)):
                cleaned_string_list.append(string) 
                
        return cleaned_string_list


    def _calc_modified_levenshtein_distance(self, lemma, non_lemma):
        if (len(lemma) > len(non_lemma) or abs(len(lemma) - len(non_lemma)) > 3):
            return -1
        
        prefix_match_count = 0
        for idx, c in enumerate(lemma):
            if (lemma[idx] == non_lemma[idx]):
                prefix_match_count += 1
                
        if (prefix_match_count < 0.75 * len(lemma)):
            return -1
        
        distances = np.zeros((len(lemma) + 1, len(non_lemma) + 1))

        for t1 in range(len(lemma) + 1):
            distances[t1][0] = t1
        for t2 in range(len(non_lemma) + 1):
            distances[0][t2] = t2
        
        for t1 in range(1, len(lemma) + 1):
            for t2 in range(1, len(non_lemma) + 1):
                if (lemma[t1 - 1] == non_lemma[t2 - 1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]
                    distances[t1][t2] = 1 + min(a, b, c)

        return int(distances[len(lemma)][len(non_lemma)])



class WikiLemmatizer(BaseWikiLemmatizer):

    parsed_file_path = 'data/parsed.csv'
    cleaned_file_path = 'data/cleaned.csv'


    def _parse_data(self, input_file_path, output_file_path):
        expression_1 = '\[\[([^|\]]+)\|?([^|\]]+)?\]\]([a-zA-ZáäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]+)?'
        expression_2 = '\(.*\)'

        with open(input_file_path, 'r', encoding='UTF-8') as input_file, open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for line in input_file:  
                # parse link and anchor text
                match_list = re.findall(expression_1, line)
                
                for match in match_list:
                    link = match[0]
                    # remove disambiguation text from link
                    link = re.sub(expression_2, '', link)
                    anchor_text = match[1]
                    anchor_text_substring = match[2]
            
                    if (anchor_text != ''):
                        output_file.write(f'{link}|{anchor_text}\n')
                    elif (anchor_text_substring != ''):
                        anchor_text = link + anchor_text_substring
                        output_file.write(f'{link}|{anchor_text}\n')
                    else:
                        output_file.write(f'{link}\n')


    def _clean_data(self, input_file_path, output_file_path):
        expression = '[^a-zA-ZáäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ|]+'

        with open(input_file_path, 'r', encoding='UTF-8') as input_file, open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for line in input_file:
                line = html.unescape(line)
                line = line.replace('&nbsp;', ' ')
                line = line.replace('&amp;', ' ')
                line = re.sub(expression, ' ', line)
                line = line.strip()
                
                if (self._check_if_string_contains_any_letters(line)): 
                    output_file.write(line + '\n')


    def _tokenize_and_lemmatize_data(self, input_file_path, output_file_path):
        max_word_count = 3
        miss_word_count = 0
        stop_words = set(line.strip() for line in open('stop_words.txt', 'r', encoding='UTF-8'))
        
        with open(input_file_path, 'r', encoding='UTF-8') as input_file, open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for line in input_file:
                link, anchor_text = self._get_link_and_anchor_text(line, '|')    
                link_list = link.split()
                link_list = self._clean_tokenized_string_list(link_list, stop_words)
                
                anchor_text_list = [] 
                if (anchor_text):
                    anchor_text_list = anchor_text.split()
                    anchor_text_list = self._clean_tokenized_string_list(anchor_text_list, stop_words)
                
                if (len(link_list) > 0):                     
                    if (len(anchor_text_list) > 0):
                        if (len(link_list) != len(anchor_text_list)):
                            miss_word_count += 1
                            
                        for idx, lemma in enumerate(link_list):
                            if ((idx + 1) > max_word_count):
                                break
                            
                            is_levenshtein_distance_matched = False
                            
                            # get all non-lemmas
                            for non_lemma in anchor_text_list:
                                levenshtein_distance = self._calc_modified_levenshtein_distance(lemma, non_lemma)
                            
                                if (levenshtein_distance > 0 and levenshtein_distance <= 3):
                                    is_levenshtein_distance_matched = True
                                    output_file.write(lemma + '|' + non_lemma + '\n')

                            if (is_levenshtein_distance_matched == False):
                                output_file.write(lemma + '\n')
                    else:
                        for idx, lemma in enumerate(link_list):
                            if ((idx + 1) > max_word_count):
                                break
                            
                            output_file.write(lemma + '\n')
                    
        return miss_word_count


    def lemmatize(self, input_file_path, output_file_path):
        print('Starting lemmatization process')

        self._parse_data(input_file_path, self.parsed_file_path)
        print('Parsing process has finished')

        self._clean_data(self.parsed_file_path, self.cleaned_file_path)
        print('Cleaning process has finished')

        miss_word_count = self._tokenize_and_lemmatize_data(self.cleaned_file_path, output_file_path) 
        print('Lemmatization process has finished')
        print(f'The number of records for which the number of words in the link and the anchor text did not match: {miss_word_count}')



class PysparkWikiLemmatizer(BaseWikiLemmatizer):

    def _parse_line(self, line):
        expression_1 = '\[\[([^|\]]+)\|?([^|\]]+)?\]\]([a-zA-ZáäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]+)?'
        expression_2 = '\(.*\)'
        match_list = re.findall(expression_1, line)
        parsed_match_list = []

        for match in match_list:
            link = match[0]
            # remove disambiguation text from link
            link = re.sub(expression_2, '', link)
            anchor_text = match[1]
            anchor_text_substring = match[2]

            if (anchor_text != ''):
                parsed_match_list.append(link + '|' + anchor_text)
            elif (anchor_text_substring != ''):
                anchor_text = link + anchor_text_substring
                parsed_match_list.append(link + '|' + anchor_text)
            else:
                parsed_match_list.append(link)
                
        return parsed_match_list


    def _clean_line(self, line):
        expression = "[^a-zA-ZáäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ|]+"
        
        line = html.unescape(line)
        line = line.replace('&nbsp;', ' ').replace('&amp;', ' ')
        line = re.sub(expression, ' ', line)
        line = line.strip()
        
        return line


    def _parse_and_clean_data(self, spark_session, input_file_path):
        expression = '\[\[([^|\]]+)\|?([^|\]]+)?\]\]([a-zA-ZáäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]+)?'

        parsed_rdd = spark_session.read.format('com.databricks.spark.xml') \
            .option('rowTag', 'page') \
            .load(input_file_path) \
            .rdd.map(lambda row_obj: str(row_obj.revision.text._VALUE) if row_obj.revision and row_obj.revision.text else '') \
            .filter(lambda line: re.search(expression, line)) \
            .flatMap(lambda line: self._parse_line(line)) \
            .map(lambda line: self._clean_line(line)) \
            .filter(lambda line: self._check_if_string_contains_any_letters(line))
            
        return parsed_rdd


    def _tokenize_and_lemmatize_line(self, line, stop_words):
        lemmatized_list = []
        link, anchor_text = self._get_link_and_anchor_text(line, '|')    
        max_word_count = 3
        
        link_list = link.split()
        link_list = self._clean_tokenized_string_list(link_list, stop_words)
        
        anchor_text_list = [] 
        if (anchor_text):
            anchor_text_list = anchor_text.split()
            anchor_text_list = self._clean_tokenized_string_list(anchor_text_list, stop_words)
            
        if (len(link_list) > 0):                     
            if (len(anchor_text_list) > 0):
                for idx, lemma in enumerate(link_list):
                    if ((idx + 1) > max_word_count):
                        break

                    is_levenshtein_distance_matched = False

                    # get all non-lemmas
                    for non_lemma in anchor_text_list:
                        levenshtein_distance = self._calc_modified_levenshtein_distance(lemma, non_lemma)

                        if (levenshtein_distance > 0 and levenshtein_distance <= 3):
                            is_levenshtein_distance_matched = True
                            lemmatized_list.append(lemma + '|' + non_lemma)

                    if (is_levenshtein_distance_matched == False):
                        lemmatized_list.append(lemma)
            else:
                for idx, lemma in enumerate(link_list):
                    if ((idx + 1) > max_word_count):
                        break

                    lemmatized_list.append(lemma)
        
        return lemmatized_list


    def _tokenize_and_lemmatize_data(self, cleaned_rdd): 
        stop_words = set(line.strip() for line in open('stop_words.txt', 'r', encoding='UTF-8'))

        lemmatized_rdd = cleaned_rdd.flatMap(lambda line: self._tokenize_and_lemmatize_line(line, stop_words)) \
            .filter(lambda line: self._check_if_string_contains_any_letters(line))
        
        return lemmatized_rdd

    
    def _save_to_file(self, rdd, output_file_path):
        with open(output_file_path, 'w', encoding='UTF-8') as output_file:
            for element in rdd.collect():
                output_file.write(element + '\n')


    def lemmatize(self, spark_session, input_file_path, output_file_path):
        print('Starting lemmatization process')

        cleaned_rdd = self._parse_and_clean_data(spark_session, input_file_path)
        lemmatized_rdd = self._tokenize_and_lemmatize_data(cleaned_rdd)
        self._save_to_file(lemmatized_rdd, output_file_path)
        
        print('Parsing process has finished')
        print('Cleaning process has finished')
        print('Lemmatization process has finished')
            



# if __name__ == '__main__':
    # INPUT_WIKI_FILE = 'data/skwiki-latest-pages-articles.xml'
    # INPUT_WIKI_FILE = 'data/skwiki-latest-pages-articles_debug.xml'
    # INPUT_WIKI_FILE = 'data/skwiki-latest-pages-articles_debug2.xml'