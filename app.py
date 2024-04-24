import codecs
from decimal import Decimal
from flask import Flask, request, jsonify

def parse_traindata():
    fin = 'static/hmmmodel.txt'
    transition_prob = {}
    emission_prob = {}
    tag_list = []
    tag_count ={}
    word_set = set()
    tag_set = set()
    try:
        input_file = codecs.open(fin,mode ='r', encoding="utf-8")
        lines = input_file.readlines()
        flag = False
        for line in lines:
            line = line.strip('\n')
            if line != "Emission Model":
                i = line[::-1]
                key_insert = line[:-i.find(":")-1]
                value_insert = line.split(":")[-1]

                # for transition probabilities #
                if flag == False:
                    transition_prob[key_insert] = value_insert
                    if (key_insert.split("~tag~")[0] not in tag_list) and (key_insert.split("~tag~")[0] != "start"):
                        tag_list.append(key_insert.split("~tag~")[0])

                else:
                    # for emission probabilities #
                    emission_prob[key_insert] = value_insert
                    val = key_insert.split("/")[-1]
                    j = key_insert[::-1]
                    word = key_insert[:-j.find("/")-1].lower()
                    word_set.add(word)
                    if val in tag_count:
                        tag_count[val] +=1
                    else:
                        tag_count[val] = 1
                    tag_set.add(val)

            else:
                flag = True
                continue

        input_file.close()
        return tag_list, transition_prob, emission_prob, tag_count, word_set

    except IOError:
        return {'error': 'HMM not found'}

absent_word_set = set()

def viterbi_algorithm(sentence, tag_list, transition_prob, emission_prob,tag_count, word_set):
    global tag_set
    # Get words from each sentence #
    sentence = sentence.strip("\n")
    word_list = sentence.split(" ")
    current_prob = {}
    for tag in tag_list:
        # transition probability #
        tp = Decimal(0)
        # Emission probability #
        em = Decimal(0)
        # Storing the probability of every tag to be starting tag #
        if "start~tag~"+tag in transition_prob:
            tp = Decimal(transition_prob["start~tag~"+tag])
        # Check for word in training data. If present, check the probability of the first word to be of given tag#
        if word_list[0].lower() in word_set:
            if (word_list[0].lower()+"/"+tag) in emission_prob:
                em = Decimal(emission_prob[word_list[0].lower()+"/"+tag])
                # Storing probability of current combination of tp and em #
                current_prob[tag] = tp * em
         # Check for word in training data. If absent then probability is just tp# 
        else:
            absent_word_set.add(word_list[0])
            em = Decimal(1) /(tag_count[tag] +len(word_set))
            current_prob[tag] = tp

    if len(word_list) == 1:
        # Return max path if only one word in sentence #
        max_path = max(current_prob, key=current_prob.get)
        return max_path
    else:
        # Tracking from second word to last word #
        for i in range(1, len(word_list)):
            previous_prob = current_prob
            current_prob = {}
            locals()['dict{}'.format(i)] = {}
            previous_tag = ""
            for tag in tag_list:
                if word_list[i].lower() in word_set:
                    if word_list[i].lower()+"/"+tag in emission_prob:
                        em = Decimal(emission_prob[word_list[i].lower()+"/"+tag])
                        # Find the maximum probability using previous node's(tp*em)[i.e probability of reaching to the previous node] * tp * em (Bigram Model) #
                        max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(transition_prob[previous_tag + "~tag~" + tag]) * em, previous_tag) for previous_tag in previous_prob)
                        current_prob[tag] = max_prob
                        locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                        previous_tag = previous_state
                else:
                    absent_word_set.add(word_list[i])
                    em = Decimal(1) /(tag_count[tag] +len(word_set))
                    max_prob, previous_state = max((Decimal(previous_prob[previous_tag]) * Decimal(transition_prob[previous_tag+"~tag~"+tag]) * em, previous_tag) for previous_tag in previous_prob)
                    current_prob[tag] = max_prob
                    locals()['dict{}'.format(i)][previous_state + "~" + tag] = max_prob
                    previous_tag = previous_state

            # if last word of sentence, then return path dicts of all words #
            if i == len(word_list)-1:
                max_path = ""
                last_tag = max(current_prob, key=current_prob.get)
                max_path = max_path + last_tag + " " + previous_tag
                for j in range(len(word_list)-1,0,-1):
                    for key in locals()['dict{}'.format(j)]:
                        data = key.split("~")
                        if data[-1] == previous_tag:
                            max_path = max_path + " " +data[0]
                            previous_tag = data[0]
                            break
                result = max_path.split()
                result.reverse()
                return " ".join(result)
            
tag_to_meaning = {
    'NN': 'Noun',
    'NNP': 'Proper Noun',
    'JJ': 'Adjective',
    'DEM': 'Determiner / Demonstrative',
    'INJ': 'Interjection',
    'INTF': 'Adverb (intensifier)',
    'NEG': 'Negation',
    'NST': 'Spatial Nouns',
    'RP': 'Particles',
    'PRP': 'Pronoun',
    'RB': 'Adverb',
    'RDP': 'Reduplications',
    'AF': 'Quantifiers',
    'VAUX': 'Auxiliary verb',
    'SYM': 'Symbol',
    'PSP': 'Postposition',
    'CC': 'Coordination Conjunction',
    'QC': 'Cardinals',
    'QO': 'Ordinals'
}

app = Flask(__name__)

@app.route("/pos_tagger")
def pos_tagger():
    data = request.get_json() 
    tag_list, transition_model, emission_model, tag_count, word_set = parse_traindata()
    sentence = data.get('sentence') 
    result = viterbi_algorithm(sentence, tag_list, transition_model, emission_model, tag_count, word_set)
    result_array = result.split()
    result_with_meanings = ", ".join([tag_to_meaning.get(tag, tag) for tag in result_array])
    return jsonify(sentence=sentence, result=result_with_meanings)