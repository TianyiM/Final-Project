from random import shuffle



shuffle(pos_review) 
size = int(len(pos_review)/2 - 18)
pos = pos_review[:size]
neg = neg_review

def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg']
    return negFeatures

train = posFeatures[174:]+negFeatures[124:]
test = posFeatures[:124]+negFeatures[:124]
