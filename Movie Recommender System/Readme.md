# Content-Based Movie Recommendation System

## 📖 Description

This project is a content-based movie recommendation system that suggests movies to users based on their content. The core idea is to recommend items that are similar to those a user already prefers. The similarity of movies is calculated based on a collection of their metadata, combined into "tags" — such as genre, keywords, cast, and director.

For example, if a user enjoys "The Dark Knight," the system will recommend other movies that have similar tags, suggesting other superhero films or movies by the same director.

## ✨ Features

* **Content-Based Filtering:** Recommends movies by comparing their attributes (genre, cast, director, keywords).

* **Bag-of-Words Model:** Uses `CountVectorizer` to convert text-based tags into a matrix of token counts.

* **Cosine Similarity:** Calculates the similarity between movies based on their feature vectors to find the most similar items.

* **Optimized Vocabulary:** Limits the feature set to the top 5,000 most frequent words and removes common English stop words for better performance and relevance.

* **Interactive and Simple:** Provides a straightforward interface to get movie recommendations. Just enter a movie title and get a list of similar ones!

* **Scalable:** The approach can be easily adapted to include more movie attributes or larger datasets.

## 🤔 How It Works

The recommendation engine follows these steps:

1. **Data Collection & Preprocessing:** Movie metadata is loaded from the dataset. Key features like `genres`, `keywords`, `cast`, `crew` (specifically the director) are cleaned and processed.

2. **Feature Engineering:** The key attributes for each movie are combined into a single text string called `tags`. This string serves as the complete content profile for each movie.

3. **Vectorization:** The `CountVectorizer` from Scikit-learn is used to convert the `tags` for all movies into a numerical matrix. It creates a vocabulary of the 5,000 most frequent words across all tags (while ignoring common English stop words). Each movie is then represented as a vector where each element is the count of a word in its tags.

4. **Similarity Calculation:** The cosine similarity is computed between the vector of the input movie and all other movies in the dataset. This metric effectively measures the angle between the vectors, providing a similarity score from 0 to 1.

5. **Recommendation:** The system sorts the movies based on their similarity scores in descending order and returns the top N most similar movies as recommendations.

## 💾 Dataset

This project uses the TMDB 5000 Movie Dataset, which is available on Kaggle. The dataset consists of two CSV files:

* `tmdb_5000_movies.csv`: Contains information about 5000 movies, including budget, genres, homepage, keywords, original language, overview, popularity, etc.

* `tmdb_5000_credits.csv`: Contains cast and crew information for each movie, including movie ID, title, cast, and crew.

### 🚀 Installation
To get a local copy up and running, follow these simple steps.

**Prerequisites**
You need to have Python 3.x installed on your system. You can download it from python.org.

**Steps**
1. **Clone the repository:**
``
git clone https://github.com/AkibDa/ml_projects
cd Movie\ Recommender\ System/
``
2. **Install the required packages:**
It's recommended to use a virtual environment
```
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```
*Note: If a `requirements.txt` file is not available, you can install the necessary libraries manually:*
```
pip install pandas numpy scikit-learn streamlit
```
3. **Download the dataset:**
Download the TMDB 5000 dataset from the link above and place the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` files into a `data/` directory in the project root.

## 🛠️ Technologies Used

* **Python:** The core programming language.

* **Pandas:** For data manipulation and CSV file handling.

* **NumPy:** For numerical operations.

* **Scikit-learn:** For implementing CountVectorizer and Cosine Similarity.

* **Jupyter Notebook:** For interactive development and experimentation.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project

2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)

3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)

4. Push to the Branch (`git push origin feature/AmazingFeature`)

5. Open a Pull Request