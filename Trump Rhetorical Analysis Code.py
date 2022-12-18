#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Import Libraries
from datascience import *
import numpy as np
import pandas as pd
import re


# In[2]:


###Functions Galore

def word_separate(text):
    """Given a string, separate the words into a list, make them lowercase, and leave out punctuation."""
    return text.replace(",", "").replace(".", "").replace(";", "").replace("?", "").replace("!", "").replace(":", "").replace("(", "").replace(")", "").replace("(inaudible)", "").split(" ")

def capital_separate(li):
    """Given a list, separate any elements of that list that are two words, joined by a capital letter
    that starts the second word: youOklahoma --> you, Oklahoma or PresidentThank --> President, Thank.
    Then put those separated words in their proper place in the list."""
    split_word_list = []
    for i in np.arange(len(li)):
        word = li[i]
        split_word = re.findall("[a-zA-Z][^A-Z]*", word)
        split_word_list.extend(split_word)
    return split_word_list

def lowercase(array):
    """Given a list, lowercase all of the elements within the list"""
    lower_words = make_array()
    for i in np.arange(len(array)):
        lower_word = array.item(i).lower()
        lower_words = np.append(lower_words, lower_word)
    return lower_words

def count_repetitions(array):
    """Given an array, count the number of elements that are the same, outputting an array with the number of times
    the respective word is in the original list"""
    counts = make_array()
    li = array.tolist()
    for i in np.arange(len(array)):
        count = li.count(li[i])
        counts = np.append(counts, count)
    return counts

def df_converter(df):
    """Given a dataframe with columns "Word" and "Count", output a numpy table with the same columns"""
    return Table().with_columns("Word", np.array(df["Word"].values.tolist()), "Count", np.array(df["Count"].values.tolist()))

def remove_articles(table):
    """Given a numpy table with columns "Word" and "Count", remove any rows with "Word" == an article"""
    return table.where("Word", are.not_equal_to("a")).where("Word", are.not_equal_to("an")).where("Word", are.not_equal_to("the"))

def remove_conjunctions(table):
    """Given a numpy table with columns "Word" and "Count", remove any rows with "Word" == a conjunction"""
    return table.where("Word", are.not_equal_to("for")).where("Word", are.not_equal_to("and")).where("Word", are.not_equal_to("nor")).where("Word", are.not_equal_to("but")).where("Word", are.not_equal_to("or")).where("Word", are.not_equal_to("yet")).where("Word", are.not_equal_to("so"))

def remove_demonstratives(table):
    """Given a numpy table with columns "Word" and "Count", remove any rows with "Word" == a demonstrative pronoun"""
    return table.where("Word", are.not_equal_to("this")).where("Word", are.not_equal_to("that")).where("Word", are.not_equal_to("these")).where("Word", are.not_equal_to("those")).where("Word", are.not_equal_to("such"))

def remove_prepositions(table):
    """Given a numpy table with columns "Word" and "Count", remove any rows with "Word" == a common preposition"""
    return table.where("Word", are.not_equal_to("to")).where("Word", are.not_equal_to("of")).where("Word", are.not_equal_to("in")).where("Word", are.not_equal_to("as"))

def remove_all(table):
    """Given a numpy table, run all above functions on the table"""
    return remove_articles(remove_conjunctions(remove_prepositions(remove_demonstratives(table))))

def remove_pronouns(table):
    """Given a numpy table with columns "Word" and "Count", remove any rows with "Word" == a pronoun"""
    return table.where("Word", are.not_equal_to("i")).where("Word", are.not_equal_to("me")).where("Word", are.not_equal_to("my")).where("Word", are.not_equal_to("mine")).where("Word", are.not_equal_to("you")).where("Word", are.not_equal_to("your")).where("Word", are.not_equal_to("yours")).where("Word", are.not_equal_to("he")).where("Word", are.not_equal_to("him")).where("Word", are.not_equal_to("his")).where("Word", are.not_equal_to("she")).where("Word", are.not_equal_to("her")).where("Word", are.not_equal_to("hers")).where("Word", are.not_equal_to("it")).where("Word", are.not_equal_to("its")).where("Word", are.not_equal_to("we")).where("Word", are.not_equal_to("our")).where("Word", are.not_equal_to("ours")).where("Word", are.not_equal_to("y'all")).where("Word", are.not_equal_to("ya'll")).where("Word", are.not_equal_to("they")).where("Word", are.not_equal_to("them")).where("Word", are.not_equal_to("their"))

def only_pronouns(table):
    """Given a numpy table with columns "Word" and "Count", keep any rows with "Word" == a pronoun"""
    new_table = Table().with_columns("Word", make_array(), "Count", make_array())
    return new_table.append(table.where("Word", are.equal_to("i"))).append(table.where("Word", are.equal_to("me"))).append(table.where("Word", are.equal_to("my"))).append(table.where("Word", are.equal_to("mine"))).append(table.where("Word", are.equal_to("you"))).append(table.where("Word", are.equal_to("your"))).append(table.where("Word", are.equal_to("yours"))).append(table.where("Word", are.equal_to("he"))).append(table.where("Word", are.equal_to("him"))).append(table.where("Word", are.equal_to("his"))).append(table.where("Word", are.equal_to("she"))).append(table.where("Word", are.equal_to("her"))).append(table.where("Word", are.equal_to("hers"))).append(table.where("Word", are.equal_to("it"))).append(table.where("Word", are.equal_to("its"))).append(table.where("Word", are.equal_to("we"))).append(table.where("Word", are.equal_to("our"))).append(table.where("Word", are.equal_to("ours"))).append(table.where("Word", are.equal_to("y'all"))).append(table.where("Word", are.equal_to("ya'll"))).append(table.where("Word", are.equal_to("they"))).append(table.where("Word", are.equal_to("them"))).append(table.where("Word", are.equal_to("their")))

def word_frequency(table):
    """Given a numpy table with columns "Word" and "Count", return the relative frequency of the counts"""
    sum_total = np.sum(table.column(1).astype(np.float))
    frequency = make_array()
    for i in np.arange(len(table.column(1))):
        if table.column(1).astype(np.float).item(i) == 0:
            frequency = np.append(frequency, 0)
        else:
            frequency = np.append(frequency, table.column(1).astype(np.float).item(i) / sum_total)
    return table.with_column("Relative Frequency", frequency)

def frequency_difference(table1, table2):
    """Given two tables, calculate the difference in frequency between each of the words, outputting a table"""
    table1_frequency = word_frequency(table1)
    table2_frequency = word_frequency(table2).relabeled("Relative Frequency", "Relative Frequency 2")
    table1_data = {"Word": table1_frequency.column(0).tolist(), "Count": table1_frequency.column(1).tolist(), "Relative Frequency": table1_frequency.column(2).tolist()}
    table1_df = pd.DataFrame(table1_data, columns = ["Word", "Count", "Relative Frequency"])
    table2_data = {"Word": table2_frequency.column(0).tolist(), "Count 2": table2_frequency.column(1).tolist(), "Relative Frequency 2": table2_frequency.column(2).tolist()}
    table2_df = pd.DataFrame(table2_data, columns = ["Word", "Count 2", "Relative Frequency 2"])
    merged_df = pd.merge(table1_df, table2_df, on="Word", how="outer")
    merged_table = Table().with_columns("Word", np.array(merged_df["Word"].values.tolist()), "Count", np.array(merged_df["Count"].values.tolist()), "Relative Frequency", np.array(merged_df["Relative Frequency"].values.tolist()), "Count 2", np.array(merged_df["Count 2"].values.tolist()), "Relative Frequency 2", np.array(merged_df["Relative Frequency 2"].values.tolist()))
    frequency_table = merged_table.with_columns("Frequency Difference", merged_table.column("Relative Frequency 2") - merged_table.column("Relative Frequency"), "Frequency Percent Difference", merged_table.column("Relative Frequency 2") / merged_table.column("Relative Frequency"))
    return frequency_table


# In[3]:


###Clean Joe Biden Data
#Convert table into one lowercase string
biden_speech_table = Table.read_table("joe_biden_dnc_2020.csv")
biden_speech_string = ""
for i in np.arange(biden_speech_table.num_rows):
    biden_speech_string = biden_speech_string + biden_speech_table[0][i].lower() + " "
biden_speech_string

#Separate string into array of words
biden_speech_array = np.array(word_separate(biden_speech_string))

#Check
print(type(biden_speech_array))
print(biden_speech_array)


# In[4]:


###Clean Donald Trump RNC Data
trump_file1 = open("donald_trump_first_speech.txt")
trump_speech1_string = trump_file1.read()
trump_file1.close()

trump_file2 = open("donald_trump_second_speech.txt")
trump_speech2_string = trump_file2.read()
trump_file2.close()

#Separate strings into a single array of words
trump_speech_string = trump_speech1_string + trump_speech2_string
trump_speech_dirty = word_separate(trump_speech_string)

#Clean array of joint words like youOklahoma
trump_speech_cleaned = np.array(capital_separate(trump_speech_dirty))

#Lowercase all strings
trump_speech_array = lowercase(trump_speech_cleaned)

#Check
print(type(trump_speech_array))
print(trump_speech_array)


# In[5]:


###Clean GOP Debate 2016 Data
trump_debates_lines = []
debate_file = open("gop_debate_all.txt")
for line in debate_file:
    if 'TRUMP:' in line:
        trump_debates_lines.append(line)
debate_file.close()

#Convert string of lines into one string, cleaning lines and speaker signifier
debates_string = ""
for i in np.arange(len(trump_debates_lines)):
    debates_string = debates_string + trump_debates_lines[i]
trump_debates_string = debates_string.replace("TRUMP: ", "").replace("\n", "")

#Convert string into array
trump_debates_array_dirty = word_separate(trump_debates_string)
trump_debates_array_clean = np.array(capital_separate(trump_debates_array_dirty))

#Lowercase array
trump_debates_array = lowercase(trump_debates_array_clean)

#Check
print(type(trump_debates_array))
print(trump_debates_array)


# In[6]:


###Calculate count of words and put them in table form
#Biden
biden_counts = count_repetitions(biden_speech_array)
biden_words = {"Word": biden_speech_array.tolist(), "Count": biden_counts.tolist()}
biden_df = pd.DataFrame(biden_words, columns = ["Word", "Count"])
biden_table = biden_df.drop_duplicates()
biden = df_converter(biden_table)

#Trump Speeches
trump_speech_counts = count_repetitions(trump_speech_array)
trump_speech_words = {"Word": trump_speech_array.tolist(), "Count": trump_speech_counts.tolist()}
trump_speech_df = pd.DataFrame(trump_speech_words, columns = ["Word", "Count"])
trump_speech_table = trump_speech_df.drop_duplicates()
trump_speech = df_converter(trump_speech_table)

#Trump Debates
trump_debates_counts = count_repetitions(trump_debates_array)
trump_debates_words = {"Word": trump_debates_array.tolist(), "Count": trump_debates_counts.tolist()}
trump_debates_df = pd.DataFrame(trump_debates_words, columns = ["Word", "Count"])
trump_debates_table = trump_debates_df.drop_duplicates()
trump_debates = df_converter(trump_debates_table)


# In[7]:


###Check
print(biden.show(5))
print(trump_speech.show(5))
print(trump_debates.show(5))
biden.where("Word", are.equal_to("me"))


# In[8]:


###Analysis Tables
#Biden vs Trump Convention Pronouns
convention_pronouns = frequency_difference(only_pronouns(biden), only_pronouns(trump_speech))
convention_pronouns.sort("Frequency Percent Difference", descending = False).show(5)

#Trump at RNC vs Trump Debates without pronouns, articles, etc.
speech_debates_words_cut = frequency_difference(remove_all(remove_pronouns(trump_speech)), remove_all(remove_pronouns(trump_debates)))
speech_debates_words_cut.sort("Frequency Percent Difference", descending = False).show(5)

#Trump at RNC vs Trump Debates
speech_debates_words = frequency_difference(trump_speech, trump_debates)
speech_debates_words.sort("Frequency Percent Difference", descending = False).show(5)

#Trump Pronouns at RNC vs Trump Pronouns at Debates
speech_debates_pronouns = frequency_difference(only_pronouns(trump_speech), only_pronouns(trump_debates))
speech_debates_pronouns.sort("Frequency Percent Difference", descending = False).show(5)


# In[9]:


###Export Tables to csv
convention_pronouns.to_df().to_csv('Convention Pronouns, Biden v Trump.csv', index=False)
speech_debates_words_cut.to_df().to_csv('Words Removed, Convention v Debates.csv', index=False)
speech_debates_words.to_df().to_csv('All Words, Convention v Debates.csv', index=False)
speech_debates_pronouns.to_df().to_csv('Pronouns, Convention v Debates.csv', index=False)


# In[ ]:




