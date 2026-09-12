import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from multilingual_examples import EXAMPLES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")


def load_intents():
    """Load English, Hindi, and Marathi training examples."""
    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []
    labels = []

    for intent, examples in data["intents"].items():
        if isinstance(examples, dict):
            example_lists = examples.values()
        else:
            example_lists = [examples]
        for example_list in example_lists:
            for example in example_list:
                texts.append(example)
                labels.append(intent)

    for intent, by_lang in EXAMPLES.items():
        for example_list in by_lang.values():
            for example in example_list:
                texts.append(example)
                labels.append(intent)

    return texts, labels


def train_classifier():
    """Train the intent classifier for English, Hindi, and Marathi."""
    texts, labels = load_intents()

    classifier = Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        token_pattern=r"(?u)\b\w+\b",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                    ),
                ),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                C=10,
            ),
        ),
    ])

    classifier.fit(texts, labels)

    return classifier


def predict_intent(classifier, user_message, threshold=0.30):
    """
    Predict the intent of a user message.

    If the classifier isn't confident about any label
    (max probability below `threshold`), return UNKNOWN
    instead of forcing a low-confidence guess.

    threshold=0.30 is calibrated against LogisticRegression(C=10):
    genuine in-scope messages typically score 0.6-0.9, while
    gibberish/out-of-scope messages typically score 0.05-0.15.
    Re-check this value with test_classifier.py if intents.json
    or the classifier's hyperparameters change.
    """
    probabilities = classifier.predict_proba([user_message])[0]
    best_index = probabilities.argmax()
    best_probability = probabilities[best_index]

    if best_probability < threshold:
        return "UNKNOWN"

    return classifier.classes_[best_index]


if __name__ == "__main__":

    classifier = train_classifier()

    print("\nSmart Agri Intent Classifier")
    print("=" * 40)
    print("Type 'exit' to stop.\n")

    while True:

        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            break

        intent = predict_intent(classifier, user_message)

        print(f"Intent: {intent}\n")