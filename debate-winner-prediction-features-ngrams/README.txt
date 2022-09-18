- A description for features and limitations:
The idea for designing these features is to represent each debate by 'how much the pro side surpassed the con side'.
Most features except for the user features are created following a strategy of subtracting features of the pro side by those of the cons side.
Features implemented in the model include:
    - Word Ngram (with n=1,2,3) representations weigted by tf-idf scores, filtering out ngrams with small or large document frequencies
    - Lexicon-based features: total V/A/D scores (sum up V/A/D scores for each word)
    - Linguistic features: number of personal pronouns and number of excalamations
    - User features: opinion tendency of the group of voters on big issues, political ideologies and religious ideologies