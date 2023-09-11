import requests
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class Review:
    def __init__(self, text):
        self.text = text


class Product:
    def __init__(self, name, description, ratings, reviews):
        self.name = name
        self.description = description
        self.ratings = ratings
        self.reviews = reviews
        self.sustainable_features = None


class EcoFriendlyProductRecommendation:
    def __init__(self):
        self.products = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def scrape_product_info(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        product_elements = soup.select(".product")
        for product_element in product_elements:
            name = product_element.select_one(".name").text.strip()
            description = product_element.select_one(
                ".description").text.strip()
            ratings = float(product_element.select_one(
                ".ratings").text.strip())
            reviews = []
            review_elements = product_element.select(".review")
            for review_element in review_elements:
                review_text = review_element.text.strip()
                reviews.append(Review(review_text))
            product = Product(name, description, ratings, reviews)
            self.products.append(product)

    def analyze_product_data(self):
        corpus = [product.description for product in self.products]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.tfidf_matrix)
        labels = kmeans.labels_
        for i, product in enumerate(self.products):
            product.sustainable_features = labels[i]

    def generate_recommendations(self, user_preferences):
        user_preferences_vector = self.tfidf_vectorizer.transform(
            [user_preferences])
        product_scores = np.dot(user_preferences_vector,
                                self.tfidf_matrix.T).toarray()[0]
        sorted_indices = np.argsort(product_scores)[::-1]
        recommendations = [self.products[i] for i in sorted_indices]
        return recommendations

    def update_product_data(self):
        # Nothing to update at the moment
        pass


def main():
    eco_friendly_recommendation = EcoFriendlyProductRecommendation()
    product_url = input("Please enter the URL for product information: ")
    eco_friendly_recommendation.scrape_product_info(product_url)
    eco_friendly_recommendation.analyze_product_data()

    user_preferences = input("Please enter your sustainability preferences: ")
    recommendations = eco_friendly_recommendation.generate_recommendations(
        user_preferences)

    for i, recommendation in enumerate(recommendations):
        print(f"\nRecommendation {i+1}:")
        print("==============================")
        print("Name:", recommendation.name)
        print("Description:", recommendation.description)
        print("Ratings:", recommendation.ratings)
        print("Reviews:")
        for review in recommendation.reviews:
            print("-", review.text)
        print("Sustainable Features:", recommendation.sustainable_features)


if __name__ == "__main__":
    main()
