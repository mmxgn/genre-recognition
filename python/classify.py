import joblib

from collections import Counter
from musicnn.extractor import extractor

# Set the musical genres you want to check against
TAGS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

# Dictionaries for accessing tags
idx2tag = dict((n, TAGS[n]) for n in range(len(TAGS)))


def classify_audio(fname: str, model_path="../models/features_classifier.pkl"):
    # Load classifier
    clf = joblib.load(model_path)

    # Extract the `max_pool' features using musicnn
    _, _, features = extractor(
        fname, model="MSD_musicnn", input_overlap=1, extract_features=True
    )

    maxpool_features = features["max_pool"]

    # Classify each feature with the trained classifier
    y_pred = clf.predict(maxpool_features)

    # Assign a string tag (e.g. 'classical' instead of 1) to each predicted class
    y_pred_labels = [idx2tag[n] for n in y_pred]

    # The assigned genre is the most common tag
    tag = Counter(y_pred_labels).most_common()[0][0]

    return tag


if __name__ == "__main__":
    """
    If called from the command line, accept 
    as arguments a .wav file to classify, and
    additionally a model path.
    """

    import argparse

    parser = argparse.ArgumentParser(
        "Simple utility that assigns a genre to a sound file."
    )

    parser.add_argument("path", help="The path to an audio file.")

    parser.add_argument(
        "--model-path",
        default="../models/features_classifier.pkl",
        help="The path to a classifier model.",
    )

    args = parser.parse_args()
    tag = classify_audio(args.path, args.model_path)

    print(tag, end="")
