import nltk
from nltk.corpus import brown
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('brown')
    nltk.download('universal_tagset')
    
    # Get common words, filter for verbs/nouns which are more likely to be signs
    words = nltk.FreqDist(brown.words())
    common_words = [w.lower() for w, f in words.most_common(5000) if w.isalpha() and len(w) > 2]
    
    # Select top 2000
    selected_words = sorted(list(set(common_words))[:2000])
    
    with open('actions.txt', 'w') as f:
        for word in selected_words:
            f.write(word + '\n')
            
    print(f"Generated actions.txt with {len(selected_words)} words.")
    
except Exception as e:
    print(f"Error generating words: {e}")
    # Fallback if download fails
    with open('actions.txt', 'w') as f:
        basic_words = ['hello', 'thanks', 'iloveyou', 'yes', 'no', 'help', 'please', 'good', 'bad', 'excuse', 'sorry']
        for i in range(2000):
            if i < len(basic_words):
                f.write(basic_words[i] + '\n')
            else:
                f.write(f'sign_{i}\n')
    print("Generated fallback actions.txt")
