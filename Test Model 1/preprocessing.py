import re
import string

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    HAS_SASTRAWI = True
except ModuleNotFoundError:
    HAS_SASTRAWI = False

# Initialize stopwords
if HAS_SASTRAWI:
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
else:
    stopwords = [
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu",
        "pada", "dalam", "karena", "atau", "saja", "juga", "ada", "sudah",
        "belum", "kita", "mereka", "saya", "kamu", "dia", "kami", "nya",
    ]

# Keep negations!
negation_words = ["tidak", "bukan", "jangan"]
stopwords = [word for word in stopwords if word not in negation_words]

# Slang dictionary
slang_dict = {
    "gk": "tidak",
    "ga": "tidak",
    "bgt": "banget",
    "yg": "yang",
    "dprii": "dpr",
    "wkwk": "lucu"
}

# Optional stemmer
if HAS_SASTRAWI:
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
else:
    stemmer = None


EMOJI_TOKEN_MAP = {
    "emo_pos": ["👍", "❤️", "❤", "🔥", "👏", "😍", "🥰", "✅"],
    "emo_neg": ["👎", "😡", "😠", "💔", "😢", "❌"],
    "emo_laugh": ["😂", "🤣"],
}


def replace_emoji_with_tokens(text: str) -> str:
    """Map common sentiment emojis to explicit tokens for ML features."""
    for token, emojis in EMOJI_TOKEN_MAP.items():
        for emoji in emojis:
            if emoji in text:
                text = text.replace(emoji, f" {token} ")
    return text

def clean_text(text, use_stemming=False):
    text = str(text).lower()

    # Convert emojis into learnable tokens before stripping symbols.
    text = replace_emoji_with_tokens(text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove emojis & punctuation
    text = text.encode("ascii", "ignore").decode("ascii")
    punctuation_without_underscore = string.punctuation.replace("_", "")
    text = text.translate(str.maketrans("", "", punctuation_without_underscore))
    
    # Normalize slang
    words = text.split()
    words = [slang_dict[word] if word in slang_dict else word for word in words]

    # Keep only known emoji tokens from mapped symbols.
    valid_emoji_tokens = {"emo_pos", "emo_neg", "emo_laugh"}
    words = [word for word in words if word.isalnum() or word in valid_emoji_tokens]
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    
    text = " ".join(words)
    
    # Optional stemming
    if use_stemming and stemmer is not None:
        text = stemmer.stem(text)
    
    return text