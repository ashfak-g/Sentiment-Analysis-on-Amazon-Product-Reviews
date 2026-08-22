"""
Text preprocessing module for Amazon Product Review Sentiment Analysis.
Performs text cleaning, normalization, stop word removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils import logger

class TextPreprocessor:
    """Production NLP Text Preprocessor."""

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        self._download_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        # Retain negation words as they carry strong sentiment information
        negation_words = {'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot'}
        self.stop_words = self.stop_words - negation_words
        
        self.lemmatizer = WordNetLemmatizer()

    def _download_nltk_resources(self) -> None:
        """Ensures required NLTK resources are available locally."""
        resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
        for res in resources:
            try:
                nltk.data.find(f'corpora/{res}') if res != 'punkt' else nltk.data.find(f'tokenizers/{res}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {res}")
                nltk.download(res, quiet=True)

    def clean_text(self, text: str) -> str:
        """Applies full cleaning pipeline to a single text string."""
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Strip HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # 3. Strip URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # 4. Remove punctuation & special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 5. Remove numbers
        text = re.sub(r'\d+', ' ', text)
        
        # 6. Tokenize & normalize whitespace
        tokens = text.split()
        
        # 7. Remove stop words if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            
        # 8. Lemmatize if configured
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
        cleaned_text = " ".join(tokens)
        return cleaned_text

    def transform(self, texts: list[str]) -> list[str]:
        """Transforms a batch of raw text strings."""
        return [self.clean_text(t) for t in texts]

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_text = "This app is wonderful!! <br> I loved the TNT bombs & 100% realistic pigs, cannot stop playing!"
    cleaned = preprocessor.clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned: ", cleaned)
