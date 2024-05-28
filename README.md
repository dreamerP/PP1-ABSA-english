# Intership-1-ABSA-english
# Aspect-Based Sentiment Analysis for English and using rules and lexicons

This repository presents an unsupervised method for aspect-based sentiment analysis. The method utilizes rules and various metrics from WordNet to calculate domain similarity for aspect extraction, and VADER for polarity detection.

## Overview

Aspect-based sentiment analysis involves identifying specific aspects or features within text data and determining the sentiment associated with each aspect. Our method combines linguistic rules and semantic similarity metrics to extract aspects relevant to the domain under analysis.

## Key Features

- **Unsupervised Approach:** Our method does not rely on labeled data for training, making it suitable for domains with limited annotated datasets.
  
- **Domain-Specific Analysis:** By leveraging domain similarity metrics from WordNet, our method adapts to different domains, ensuring robustness across various types of text data.

- **Integration of VADER:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is employed for polarity detection, providing accurate sentiment analysis capabilities.

## Evaluation

The proposed solution was evaluated using the dataset provided by Hu & Liu (2004) focusing on cameras and cell phones. We explored various alternatives for aspect extraction and polarity determination. The results show promising performance, surpassing existing proposals in the literature.

## Usage

To use our method, simply clone this repository and follow the instructions provided in the documentation.

## Acknowledgements

We would like to thank the authors of the datasets used in the evaluation for their valuable contributions.

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the functionality of our method.

## License

This project is licensed under the MIT License.
