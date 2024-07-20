import argparse
import json
import torch
import numpy as np
import nltk
from transformers import BertTokenizer, BertForMaskedLM
from nltk.stem import PorterStemmer

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, input_type_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_sentence_to_token(sentence, tokenizer):
    tokenized_text = tokenizer.tokenize(sentence.lower())
    nltk_sent = nltk.word_tokenize(sentence.lower())
    return tokenized_text, nltk_sent

def convert_token_to_feature(tokens, seq_length, tokenizer, prob_mask):
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_type_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = seq_length - len(input_ids)
    input_ids += [0] * padding_length
    input_mask += [0] * padding_length
    input_type_ids += [0] * padding_length
    return InputFeatures(input_ids, input_mask, input_type_ids)

def get_score(sentence, tokenizer, model, device):
    tokenize_input = tokenizer.tokenize(sentence)
    tokenize_input = ['[CLS]'] + tokenize_input + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokenize_input)
    input_ids = torch.tensor([input_ids]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    prediction_scores = outputs.logits
    return prediction_scores

def BERT_candidate_generation(source_word, pre_tokens, num_selection=10):
    ps = PorterStemmer()
    source_stem = ps.stem(source_word)
    cur_tokens = [token for token in pre_tokens if ps.stem(token) != source_stem and token != source_word]
    return cur_tokens[:num_selection]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default=None, type=str, required=True, help="The evaluation data file (JSON).")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True, help="Bert pre-trained model.")
    parser.add_argument("--output_SR_file", default=None, type=str, required=True, help="The output file for writing substitution selection.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length.")
    parser.add_argument("--num_selections", default=10, type=int, help="Total number of selections.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)

    with open(args.eval_file, 'r') as file:
        eval_data = json.load(file)

    output_sr_file = open(args.output_SR_file, "w")

    for example in eval_data:
        abstract = example['Abstract']
        sentences = nltk.sent_tokenize(abstract)
        
        simplified_sentences = []
        
        for sentence in sentences:
            tokens, words = convert_sentence_to_token(sentence, tokenizer)
            feature = convert_token_to_feature(tokens, args.max_seq_length, tokenizer, 0.5)

            tokens_tensor = torch.tensor([feature.input_ids]).to(device)
            attention_mask = torch.tensor([feature.input_mask]).to(device)
            token_type_ids = torch.tensor([feature.input_type_ids]).to(device)

            with torch.no_grad():
                outputs = model(input_ids=tokens_tensor, attention_mask=attention_mask, token_type_ids=token_type_ids)
            prediction_scores = outputs.logits

            top_k = prediction_scores[0].topk(args.num_selections)
            pre_tokens = tokenizer.convert_ids_to_tokens(top_k.indices[0].tolist())

            source_word = words[0]  # Placeholder for source word, you can implement a logic to identify complex words
            cgBERT = BERT_candidate_generation(source_word, pre_tokens, args.num_selections)

            # For simplicity, just join candidates as a string. This should be extended for actual use.
            simplified_sentences.append(' '.join(cgBERT))

        simplified_abstract = ' '.join(simplified_sentences)
        output_sr_file.write(f"Original Abstract: {abstract}\n")
        output_sr_file.write(f"Simplified Abstract: {simplified_abstract}\n\n")

    output_sr_file.close()

if __name__ == "__main__":
    main()
