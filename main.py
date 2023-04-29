import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('brown')
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import nltk
from flashtext import KeywordProcessor
import textwrap

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

def set_seed(seed: int):
    # four different random number generators are initialized 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

# Generate Summary from text
def summary_generator(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

# This function extract the noun from the text 
def get_nouns_from_text(text):
    nouns = []
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    for token in tagged_tokens:
        if token[1].startswith('N'): 
            nouns.append(token[0])
    return nouns
# This function extract get keywords 
def get_keywords_from_text(originaltext,summarytext):
  keywords = get_nouns_from_text(originaltext)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords[:4]

# Generate Question from text 
def generate_questions(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)

  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]

  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question

# Main Function 
def mainfunction(comphrehension):
    text= comphrehension
    summarized_text = summary_generator(text,summary_model,summary_tokenizer)
    set_seed(42)
    nltk.download('averaged_perceptron_tagger')
    imp_keywords = get_keywords_from_text(text,summarized_text)
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_model = question_model.to(device)
    print ("\n\n\n\n\n\n\n")
    Modelgenerated=""
    count=1
    for answer in imp_keywords:
        ques = generate_questions(summarized_text,answer,question_model,question_tokenizer)
        print(ques)
        Modelgenerated+="Question "+str(count)+": "+ques+"\n"
        Modelgenerated+="Answer "+str(count)+": "+answer.capitalize()+"\n"
        count+=1
    return Modelgenerated