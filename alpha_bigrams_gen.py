import os
from collections import Counter
import matplotlib.pyplot as plt


def display_histogram(bigram_freq):
    """
    Displays a histogram of the top N bigrams by frequency.

    Parameters:
    - bigram_freq: List of tuples (bigram, frequency)
    - top_n: Number of top bigrams to display
    """
    # Sort by frequency in descending order and select top N
    sorted_bigrams = sorted(bigram_freq, key=lambda x: x[1], reverse=True)
    
    bigrams = [item[0] for item in sorted_bigrams]
    frequencies = [item[1] for item in sorted_bigrams]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(bigrams, frequencies, color="skyblue")
    plt.xlabel("Bigrams")
    plt.ylabel("Frequency")
    plt.title("Bigrams Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("alpha_words_bigrams.png")


def read_word_list():
    with open(os.path.join(os.getcwd(), "words_alpha.txt")) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def alpha_word_bigrams(m=75):
    count = 0
    bigrams = []
    words = read_word_list()
    bigram_frequencies = sorted_bigrams_frequency(words)
    # print(bigram_frequencies)
    # input("Frequencies")
    for bigram, _ in bigram_frequencies:
        if count == m:
            return bigrams
        bigrams.append(bigram)
        count += 1


def generate_bigrams(words):
    """Generate bigrams from a list of words and count their frequency."""
    bigrams = []
    for word in words:
        word = str(word)
        bigrams.extend([word[i : i + 2] for i in range(len(word) - 1)])
    return Counter(bigrams)


def sorted_bigrams_frequency(words):
    """Sort bigrams by frequency from highest to lowest."""
    bigram_counts = generate_bigrams(words)
    return sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    count = 0
    bigrams = []
    words = read_word_list()
    bigram_frequencies = sorted_bigrams_frequency(words)
    display_histogram(bigram_frequencies)
