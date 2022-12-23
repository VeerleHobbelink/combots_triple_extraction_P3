from src.model_transformer.run_transformer_pipeline import AlbertTripleExtractor
from src.evaluation.benchmark_evaluation.metrics import classification_report
from pathlib import Path
import json
import spacy


def load_examples(path):
    """ Load examples in the form of (str: dialogue, list: triples).

    :param path: Path to test file (e.g. 'test_examples/test_full.txt')
    :return:     List of (str: dialogue, list: triples) pairs.
    """
    # Extract lines corresponding to dialogs and triples
    samples = []
    with open(path, 'r', encoding='utf-8') as file:
        block = []
        for line in file:
            if line.strip() and not line.startswith('#'):
                block.append(line.strip())
            elif block:
                samples.append(block)
                block = []
        samples.append(block)

    # Split triple arguments
    examples = []
    for block in samples:
        if len(block) == 0: 
            break
        dialog = block[1]
        triples = [string_to_triple(triple) for triple in block[2:]]
        examples.append((dialog, triples))
    
    return examples


def save_results(result, model, test_file, confidence):
    outfile = 'src/evaluation/results/{}_{}_k={}.json'.format(Path(test_file).stem, model, confidence)
    with open(outfile, 'w', encoding='utf-8') as file:
        dct = {'precision': result[0], 'recall': result[1], 'f1': result[2], 'auc': result[3], 'pr-curve': result[4]}
        json.dump(dct, file)


def string_to_triple(text_triple):
    """ Tokenizes triple line into individual arguments

    :param text_triple: plain-text string of triple
    :return:            triple of the form (subj, pred, obj, polar)
    """
    return tuple([x.strip() for x in text_triple.split(',')])


def lemmatize_triple(subj, pred, obj, polar, nlp):
    """ Takes in a triple and perspective and lemmatizes/normalizes the predicate.
    """
    pred = ' '.join([t.lemma_ for t in nlp(pred)])
    return subj, pred, obj, polar

def unique_to_abstract(pred, abstract_dict):
  """ Maps the predicate to its abstract version
      returns the abstracted predicate 
  """
  for items in list(abstract_dict.keys()): # go over all keys
    if pred in list(abstract_dict[items]) or pred == items: # if item is in the value of the keys 
      abstracted = items # it is the abstracted form 
      return abstracted

def cleaning(list_predicates):
    """
    Cleans a list of predicates 
    """
    nlp = spacy.load('en_core_web_sm')
    cleaned = set()

    for predicate in list(list_predicates):
        working = nlp(predicate)
        subset = []

        for token in working:
        # don't include punctuation marks 
            if token.pos_ == "PUNCT":
                continue

            subset.append(token.lemma_)

            # stem all the shortened versions of to be and to have 
            if len(subset) >= 2:
                for i, sep in enumerate(subset):
                # change 'm 
                    if (subset[-1]  == "m") & (subset[-2] == "'") :
                        subset.pop(-2)
                        subset[-1] = 'be'
                    # change 've 
                    elif (subset[-1]  == "ve") & (subset[-2] == "'") :
                        subset.pop(-2)
                        subset[-1] = 'have'
                
                # change 's 
            if subset[-1] == "'s" or subset[-1] == "m" or subset[-1] == "re":
                subset[-1] = 'be'
            # take out speaker since these are subj or obj 
            if subset[-1] == "SPEAKER1" or subset[-1] == "SPEAKER2" or subset[-1] == "speaker1":
                subset.pop(-1)
        cleaned.add(' '.join(subset))
    
    return cleaned

def evaluate(test_file, model, pred_dict, num_samples=-1, k=0.9, deduplication=False,  baseline = False):
    """ Evaluates the model on a test file, yielding scores for precision@k,
        recall@k, F1@k and PR-AUC.

    :param test_file:     Test file from '/test_examples'
    :param model:         Albert, Dependency or baseline model instance
    :param num_samples:   The maximum number of samples to evaluate (default: all)
    :param k:             Confidence level at which to evaluate models
    :param deduplication: Whether to lemmatize predicates to make sure duplicate predicates such as "is"
                          and "are" are removed and match across baselines (default: True)
    :return:              Scores for precision@k, recall@k, F1@k and PR-AUC
    """
    # Extract dialog-triples pairs from annotations
    examples = load_examples(test_file)
    if num_samples > 0:
        examples = examples[:num_samples]

    # Predictions
    true_triples = []
    pred_triples = []
    for i, (dialog, triples) in enumerate(examples):
        # Print progress
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), dialog))

        # Predict triples
        extractions = model.extract_triples(dialog,  verbose=True)

        # Check for error in test set formatting
        error = False
        for triple in triples:
            if len(triple) != 4:
                print('#######\nERROR\n#######')
                print(triple)
                error = True

        print('expected:', triples)
        print('found:   ', [t for c, t in extractions if c > k])

        if not error:
            true_triples.append(triples)

            # Do this for the baseline model
            if baseline:
                new = []
                for i in range(len(extractions)):
                    current = list(extractions[i][1])
                    current[1] = unique_to_abstract(current[1], pred_dict)
                    new.append(tuple([extractions[i][0], tuple(current)]))
                pred_triples.append(new)
            else:
                pred_triples.append(extractions)


    # If lemmatize is enabled, map word forms to lemmas
    if deduplication:
        print('\nPerforming de-duplication')
        nlp = spacy.load('en_core_web_sm')
        true_triples = [set([lemmatize_triple(*triple, nlp) for triple in lst]) for lst in true_triples]
        pred_triples = [set([(conf, lemmatize_triple(*triple, nlp)) for conf, triple in lst]) for lst in pred_triples]

    # Compute performance metrics
    return classification_report(true_triples, pred_triples, k=k)


if __name__ == '__main__':
    MODEL = 'albert'
    TEST_FILE = 'src/evaluation/testset_test_declarative_statements.txt'  #'src/evaluation/testset_test_single_utterances.txt'
    MIN_CONF = 0.1

    bio_lookup = {3: '', 5: 'think', 7: 'be from', 9: 'have', 11: 'be greater than', 13: 'take advantage of', 15: 'make ', 17: 'bring', 19: 'be', 21: 'get off', 23: 'recycle', 25: 'geocache', 27: 'ride', 29: 'keep from ', 31: 'gameplay', 33: 'play (game)', 35: 'stay', 37: 'hear', 39: 'want', 41: 'be ', 43: 'get', 45: 'be at', 47: 'winning/losing', 49: 'literature', 51: 'travel', 53: 'go', 55: 'do .. For', 57: 'future actions', 59: 'add', 61: 'take', 63: 'can', 65: 'be with', 67: 'think (about)', 69: 'start', 71: 'get into', 73: 'intention ', 75: 'teach', 77: 'smoke', 79: 'like ', 81: 'have something in ', 83: 'come back', 85: 'look', 87: 'try to', 89: 'cannot ', 91: 'food actions ', 93: 'relax', 95: 'personal hygiene', 97: 'marriage', 99: 'stop', 101: 'buy', 103: 'help', 105: 'be in ', 107: 'join ', 109: 'ace', 111: 'come in ', 113: 'need', 115: 'try out', 117: 'pay ', 119: 'live', 121: 'do ', 123: 'stop ', 125: 'consider', 127: 'mow', 129: 'ai', 131: 'be by ', 133: 'be for', 135: 'be about', 137: 'save (money)', 139: 'hurt', 141: 'sing', 143: 'like', 145: 'be of', 147: 'break up with', 149: 'quit', 151: 'not be able ', 153: 'name', 155: 'go on', 157: 'use', 159: 'be on '}

    abstr_pred_dict = {'': [''], 'think': ['forget', 'glad', 'hope', 'imagine', 'kid', 'know', 'mean', 'remember', 'say', 'scare', 'sort', 'spell anything wrong', 'think'], 'be from': ['call from', 'different from', 'exhausted from', 'from', 'graduate from', 'import from', 'migrate from', 'retire from', 'switch from', 'tired from', 'visit from'], 'have': ['have', 'have any room available', 'have catch', 'have experience', 'have family', 'have no problem with that', 'like have people around', 'may have', 'should have', 'should sms over', 'will have', 'will never have'], 'be greater than': ['fast than', 'hard than', 'make more than', 'well than'], 'take advantage of': ['afraid of', 'deserve of', 'get enough of', 'have dream of go', 'hear of', 'in need of', 'inherit', 'involve', 'its', 'lose track of', 'make of', 'present', 'speak of', 'take advantage of', 'take care of', 'will spend most of the weekend with'], 'make ': ['bring it over', 'could make it as', 'could try it on', 'it', 'like it cut', 'make it', 'make it a secret between we', 'read it often', 'want it', 'wear it much', 'wear it that way', 'will make it work', 'fit right', 'look', 'make', 'make bank', 'make look', 'would make'], 'bring': ['bring', 'build', 'come', 'grow', 'hold', 'raise', 'reach', 'share', 'shed', 'turn'], 'be': ['be a camera on', 'be a copy machine on', 'be good at work on', 'be on', 'be plan on go', 'be set on', 'be still on', 'must not sit on', 'spend all be money on shopping', 'try they on be travel', 'be a serious disturbance in', 'be a story about breakdance in', 'be go in', 'be in', 'be in attendance at', 'be in between', 'be in the chair', 'be in to', 'be it in', 'be more in to', 'be still in', 'be they in', 's'], 'get off': ['fight people off', 'get off', 'get off on', 'play off', 'take friday off', 'tear off', 'throw it off'], 'recycle': ['recycle', 'scrap', 'waste'], 'geocache': ['code', 'geocache'], 'ride': ['chase', 'drive', 'hike', 'ride', 'swim', 'tour', 'train'], 'keep from ': ['keep', 'keep from', 'keep safe', 'would please keep'], 'gameplay': ['draw', 'match', 'race', 'replay'], 'play (game)': ['lead', 'play', 'play hoop with', 'play the cd loud', 'play with'], 'stay': ['get along with', 'hang with', 'have stay with', 'live with', 'rather stay', 'stay', 'stay home with', 'stay with'], 'hear': ['describe', 'hear', 'hear clearly', 'sound', 'speak'], 'want': ['let', 'prefer', 'should try', 'try', 'want', 'wish', 'be able to', 'be able to take the test again', 'be excited to', 'be go to', 'be go to make be life easy', 'be hope to visit', 'be look forward to', 'be ready to', 'be require to', 'be sneak away to', 'be someone come to get', 'be take the kid to', 'be to', 'be to leave nothing for a tip', 'be to play', 'be try to get', 'be up to', 'come here to be', 'do be job remotely', 'go to be', 'go to be take', 'have to be', 'like to be', 'like to buy a keepsake for be girlfriend', 'need to be', 'need to get be dance gear ready', 'send be car to be fix', 'try to be', 'use to be', 'want the money to be transfer', 'want to be', 'want to be here tomorrow', 'will just put be coat away', 'would like to be'], 'be ': ['wass'], 'get': ['catch', 'could get', 'dig', 'find', 'fix', 'get', 'get along', 'get more', 'get ready', 'get to go get', 'get we', 'give', 'have get', 'lose', 'send', 'should get'], 'be at': ['arrive at', 'available at', 'bad at', 'eat at', 'go at', 'good at', 'stay at', 'still at', 'take lot at the bar', 'teach at', 'turn right at', 'volunteer at', 'work at', 'work night at', 'work tonight at'], 'winning/losing': ['beat', 'win'], 'literature': ['book', 'copy', 'like read', 'note', 'post', 'read', 'write'], 'travel': ['cook 100 meal for', 'exchange', 'get ready for', 'here for', 'live here for', 'look for', 'make some for', 'must leave for', 'reserve', 'travel', 'travel here for', 'travel more for', 'visit'], 'go': ['can go', 'could go', 'fall', 'go', 'go for', 'go great with', 'go hike often', 'go just for', 'go next', 'go there', 'go there with', 'go thru', 'hang', 'leave', 'pass', 'run', 'rush', 's it go'], 'do .. For': ['cook for', 'distinguish for', 'fine for', 'for', 'good for', 'hard for', 'have seat for', 'leak for', 'market for', 'marry for', 'meet for', 'nervous for', 'pray for', 'ready for', 'retire for', 'save for', 'shop for', 'stand for', 'stop for', 'suitable for', 'vote for', 'work for'], 'future actions': ['call', 'miss', 'see', 'see tomorrow', 'should see', 'tell', 'wait', 'watch'], 'add': ['add', 'could add', 'mix'], 'take': ['can take', 'can take the test again', 'could take', 'may take', 'take', 'will', 'will bring back', 'will come', 'will get', 'will hold', 'will open', 'will take', 'would give'], 'can': ['can', 'can accommodate', 'can alphabetize', 'can also try', 'can catch', 'can get', 'can give', 'can handle', 'can have', 'can have weekend free', 'can help', 'can hold', 'can make', 'can only spend', 'can phone', 'can play', 'can recommend', 'can refund', 'can relax', 'can see', 'can show', 'can speak', 'can spend', 'can stay', 'can stay there for hour', 'can try', 'can use', 'may borrow', 'will suit well'], 'be with': ['align with', 'breakdance', 'busy with', 'come with', 'could start with', 'display', 'double cross', 'enjoy work with', 'frustrate with', 'have direct contact with', 'help with', 'hike with', 'interview with', 'obsess with', 'relax with', 'set here with', 'share with', 'sleep with', 'volunteer with', 'with', 'work with'], 'think (about)': ['about', 'concern about', 'daydream about', 'feel about', 'have idea about', 'hear about', 'know a lot about', 'know about', 'know much about', 'know nothing about', 'like talk about', 'open about', 'passionate about', 'talk about', 'think about', 'wanna talk about', 'worry about'], 'start': ['begin', 'finish', 'start'], 'get into': ['back into', 'bump into', 'get into', 'into', 'transfer into'], 'intention ': ['can call it in', 'can really get in touch with', 'interested in go to', 'like to go far in', 'like to live in', 'like to see in beijing', 'look in to', 'want they all in fifty', 'would never stab in the back'], 'teach': ['graduate', 'learn', 'teach', 'teach during', 'tutor'], 'smoke': ['blast', 'rain', 'smell', 'smoke', 'smoking'], 'like ': ['can listen to', 'can not afford to', 'can t seem to make', 'charge it to', 'feel connected to', 'give he to we', 'give time to do', 'have anything to', 'have no reason to distrust', 'know how to use', 'like to do', 'like to do for fun', 'like to try', 'like we to', 'love listen to', 'love to', 'love to own', 'love to sing', 'mind if walk to', 'need this weekend to', 'own', 'prefer not to', 'prefer not to bother', 'send it to', 'take they to her grave', 'try not to', 'want something to', 'want to', 'want to do', 'want to extend', 'want to pay', 'want to sign', 'want to watch a movie together', 'willing to do', 'would ever learn to', 'would love to'], 'have something in ': ['always work in', 'get a 4 . 0 in that class', 'grow well only in', 'have always work in', 'have any experience in', 'have c complication from surgery', 'have experience in', 'have family in the area', 'have soap in they', 'help in this city', 'help while in', 'in touch with', 'read the classic in college', 'study'], 'come back': ['close', 'come home from', 'drive under', 'get back from', 'get home from', 'land', 'like cash back', 'love watch bird from indoor', 'move', 'move back here from', 'move from', 'move here when', 'move here with', 'send somebody over', 'walk around'], 'look': ['get a good look at', 'get dog at the clinic', 'get real good at', 'good', 'have breakfast at a great restaurant', 'throw a milk carton at'], 'try to': ['able to', 'able to find', 'about to', 'about to fix', 'allow to', 'close to', 'come to', 'cook to', 'could speak to', 'devote more time to', 'excited to', 'expect to', 'forget to', 'glad to', 'happen to', 'hard to', 'have to', 'have to give', 'hope to', 'inspire to', 'intend to', 'listen to', 'look forward to', 'may speak to', 'mean to', 'meditate to', 'migrate to', 'move to', 'need to', 'need to make per year', 'new to', 'open to', 'pay attention to', 'pay more attention to', 'plan to', 'prefer to', 'propose to', 'read biography to', 'ready to', 'respond to', 'ride to', 'scared to', 'seem to', 'similar to', 'sound appeal to', 'speak to', 'stick to', 'take to', 'talk to', 'tell to', 'to', 'to withdraw', 'try to', 'will have to take', 'will learn to deal with', 'will request hr to', 'write to'], 'cannot ': ['accept', 'care if', 'cause', 'could', 'could not recognize', 'could not stand', 'have not make', 'have not pick', 'have not time for', 'never', 'please that', 'would', 'would like', 'would love', 'would not change', 'would not eat', 'would not mind', 'would rather', 'can always play', 'can always use', 'can not', 'can not afford', 'can not deal with', 'can not do', 'can not drink', 'can not eat', 'can not endure', 'can not find', 'can not handle', 'can not imagine', 'can not kick', 'can not stand', 'can not wait for', 'can talk about', 'could supply', 'must water', 'type', 'will not pull'], 'food actions ': ['bake', 'binge', 'cook', 'drink', 'eat', 'only eat'], 'relax': ['like', 'like watch', 'pet sitting', 'relax', 'sleep', 'sleep like', 'tired'], 'personal hygiene': ['dye', 'exfoliate', 'paint', 'wash', 'washing', 'wear'], 'marriage': ['divorce', 'marry'], 'stop': ['dial', 'drop', 'switch'], 'buy': ['buy', 'purchase', 'sell', 'shop', 'should buy', 'should sell'], 'help': ['allow', 'assist', 'can help contact', 'guide', 'help', 'manage', 'need', 'need help', 'seek'], 'be in ': ['bear in', 'believe in', 'check in', 'dance in', 'fall in', 'follow in', 'in', 'in between', 'interested in', 'live in', 'locate in', 'major in', 'park in', 'permit in', 'play in', 'play piano in', 'proficiency in', 'serve in', 'set in', 'sing in', 'start in', 'teach in', 'work in'], 'join ': ['adopt', 'follow', 'join', 'meet', 'should join', 'should meet', 'sign', 'volunteer'], 'ace': ['ace'], 'come in ': ['bring in', 'come back in', 'come in', 'find in', 'fit in', 'get in because', 'get in english 101', 'get interested in', 'go first thing in', 'keep in', 'live here in', 'make in', 'move in', 'play the piano in', 'shut down in', 'spend the day in', 'stay in', 'stay last night in', 'wait for in'], 'need': ['allow people to check out', 'get ready to', 'get ready to take', 'get to', 'go out to', 'go to', 'go to across', 'go to college for', 'go to far from', 'go to town on', 'go to transfer', 'go to watch it out of curiosity', 'have plan to go', 'like go out to', 'like go to', 'like to go', 'like to go for', 'like to go to', 'love to go to', 'love to go to the beach with', 'love to sit around', 'move back to', 'need to go back to', 'plan to go', 'should go to buy', 'want to go out to', 'want to take'], 'try out': ['burn down', 'calm down', 'check out', 'could fill out', 'cut out', 'find out', 'help out with', 'lock out', 'pass out', 'pull out', 'try out', 'wanna try out', 'work out'], 'pay ': ['ask bill if', 'can pay with', 'cancel', 'charge', 'cost', 'deposit', 'guarantee', 'must pay', 'owe', 'pay', 'pay for', 'should pay', 'would pay'], 'live': ['could live', 'live', 'live above', 'live around', 'live there', 'live there for', 'starve', 'survive', 'be fake straight for a long time', 'do for a job', 'do for a living', 'draw they all the time all over', 'have a crush on', 'have a seat available', 'have a small team under', 'know a country song about that', 'live here for a long time', 'live here for a month', 'make a recommendation for', 'need a full time job for health insurance', 'practice a lot as a kid', 'race the car for a living', 'share a one bedroom with', 'should leave for a tip', 'work a lot during', 'work here long time'], 'do ': ['portray woman as weak', 'read those as a child', 'roleplay as', 'start as', 'work', 'work as', 'work with vet as', 'can do', 'do', 'do for', 'do for fun', 'do well than', 'fancy', 'got do', 'hate', 'let do', 'like do'], 'stop ': ['wreck'], 'consider': ['change', 'consider', 'count', 'decide', 'exercise', 'interest', 'may consider', 'might', 'plan', 'recommend', 'rule', 'should', 'should appreciate', 'should call', 'should choose from', 'should consider', 'should drink more', 'should leave', 'should light', 'should really limit'], 'mow': ['mow'], 'ai': ['ai'], 'be by ': ['be lead by', 'come by', 'know by face', 'live by', 'pay by', 'send it by airmail'], 'be for': ['be always a need for', 'be around for', 'be come here for', 'be currently look for', 'be for', 'be here for', 'be it for', 'be look for', 'be lunch time for', 'be not for', 'be not look good for', 'be not much room for', 'be off for', 'be out for', 'be plenty for', 'be quiet for', 'be serious business for', 'be shop for', 'be take mom out', 'be there before', 'be there good hiking spot near', 'be this for', 'do all be shopping through', 'drive be big truck around', 'get be key out', 'live there be whole life', 'should take be shoe off', 'spend be holiday from'], 'be about': ['be', 'be 30 pound over weight', 'be a block away', 'be about', 'be also from', 'be also into', 'be always with', 'be arrive', 'be available', 'be content', 'be decide about switch', 'be employ', 'be far away from', 'be from', 'be get', 'be glad', 'be glad the week be over', 'be have', 'be into', 'be just below', 'be just walk', 'be learn', 'be like', 'be live', 'be mixed with', 'be naturally', 'be not familar with', 'be not feel', 'be not into', 'be not more expensive than', 'be not really a fan', 'be obsess with', 'be plan', 'be really into', 'be see', 'be something wrong with', 'be still into', 'be sure', 'be that consider', 'be the novel about', 'be they from', 'be together', 'be two pound over', 'be usually out', 'burn be house down', 'can be', 'do be body good', 'dress be yorkie as a lion', 'dye be hair red', 'have be', 'help be mom out', 'know it be', 'like be', 'love be with', 'must be', 'purchase be from ikea', 'will all be', 'will be', 'will be expect', 'will there be', 'would be'], 'save (money)': ['rather save', 'save', 'spend'], 'hurt': ['arrest', 'die', 'hurt', 'kill', 'rob', 'shoot'], 'sing': ['dance', 'sing', 'singe'], 'like': ['build house upon request', 'have to give 3 shoot', 'like listen to', 'like some magic to take away', 'like to', 'like to buy', 'like to cook for', 'like to depart', 'like to drink', 'like to eat at', 'like to find', 'like to have', 'like to leave on', 'like to make', 'like to meet for a drink', 'like to pay', 'like to purchase', 'like to read', 'like to return', 'like to send', 'like to sing', 'like to stay', 'like to use', 'love to live on', 'love to look at', 'order', 'use to', 'use to buy', 'use to have', 'would like to', 'would tike to buy', 'dislike', 'enjoy', 'feel', 'feel like', 'look like', 'love', 'seem', 'sound like', 'taste like'], 'be of': ['be a fan of', 'be a huge fan of', 'be a type of', 'be an employee of', 'be just off of', 'be one of', 'be part of', 'be scared of', 'become', 'need all of be supply right away'], 'break up with': ['break up with', 'can not put up with', 'catch up on', 'fill up', 'finish up', 'grow up as', 'grow up in', 'keep up all night', 'like to meet up with', 'make up', 'meet up for', 'meet up with', 'open up', 'show up for', 'stand up at', 'take up', 'wake up', 'wake up at'], 'quit': ['quit', 'retire'], 'not be able ': ['be a run at', 'be at', 'be be family at', 'be good at', 'be just relax at', 'be just relax at home in', 'be one at', 'be very sociable at the weekend', 'have so much fun at', 'like be at', 'will be at'], 'name': ['name'], 'go on': ['go on', 'go out on', 'mind go on', 'spend too much on'], 'use': ['arrange', 'include', 'locate', 'provide', 'require', 'serve', 'support', 'use'], 'be on ': ['catch on', 'church on', 'dance on', 'depend on', 'fall on', 'keen on', 'live on', 'love work on', 'on', 'plan on', 'plan on make', 'plane on watch', 'play on', 'play soccer on', 'put on', 'scald on', 'show on', 'sit on', 'work on']}

    if MODEL == 'albert':
        model = AlbertTripleExtractor('src/model_transformer/models/2022-12-14', bio_lookup)     # here it loads both models - argument extraction en scorer 
    else:
        raise Exception('model %s not recognized' % MODEL)

    result = evaluate(TEST_FILE, model, k=MIN_CONF, deduplication=False, pred_dict = abstr_pred_dict, baseline = False )

    # Save to file
    save_results(result, MODEL, TEST_FILE, MIN_CONF)
