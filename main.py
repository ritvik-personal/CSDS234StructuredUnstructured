import json
import numpy as np
import logging
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def load_data(reviews_file, business_file, users_file):
    """
    Load reviews, business, and user data from JSON files.
    """
    logging.info("Loading data files...")
    try:
        with open(reviews_file, 'r') as f:
            reviews = json.load(f)
        with open(business_file, 'r') as f:
            businesses = json.load(f)
        with open(users_file, 'r') as f:
            users = [json.loads(line) for line in f]
        logging.info("Data files loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise
    return reviews, businesses, users


def map_categories_to_types(categories):
    """
    Maps categories to types for filtering purposes.
    """
    type_mappings = {
        "Nightlife": ["Bars", "Night Clubs", "Lounges"],
        "Restaurants": ["Pizza", "Chinese", "Italian", "Mexican"],
        "Shopping": ["Grocery", "Department Stores", "Malls"],
        "Fitness": ["Gyms", "Yoga", "Martial Arts"],
        "Education": ["Colleges", "Libraries", "Tutoring"],
        "Entertainment": ["Theaters", "Museums", "Parks"],
        "Healthcare": ["Doctors", "Dentists", "Clinics"],
    }

    if not categories:  # Handle None or empty categories
        return []

    mapped_types = set()
    for category in categories.split(", "):
        for type_name, keywords in type_mappings.items():
            if category in keywords:
                mapped_types.add(type_name)
    return list(mapped_types)



def filter_businesses_by_criteria(businesses, city, query_type=None, min_rating=None):
    """
    Filters businesses by city, type, and minimum rating.
    """
    logging.info(f"Filtering businesses for city '{city}', type '{query_type}', and minimum rating {min_rating}.")
    filtered_business_ids = []

    if city not in businesses:
        logging.warning(f"City '{city}' not found in business data.")
        return []

    for category, biz_list in businesses[city].items():
        for business in biz_list:
            business_types = map_categories_to_types(business.get("categories", ""))
            if query_type and query_type not in business_types:
                continue
            if min_rating and business.get("stars", 0) < min_rating:
                continue
            filtered_business_ids.append(business["business_id"])

    logging.info(f"Found {len(filtered_business_ids)} businesses matching the criteria.")
    return filtered_business_ids


def get_friends(user_id, users):
    """
    Retrieves the friends list for a specific user ID from the users dataset.
    """
    logging.info(f"Looking up friends for user {user_id}...")
    user_data = next((user for user in users if user["user_id"] == user_id), None)
    if user_data:
        friends = user_data.get("friends", [])
        logging.info(f"Friends of user {user_id}: {friends}")  # Debugging output
        return friends
    logging.info(f"No friends found for user {user_id}.")  # Debugging output
    return []


def filter_relevant_users(reviews, business_ids, friends=None, target_user_id=None):
    """
    Filters relevant users based on business interactions, optional friends, and the target user.
    """
    logging.info("Filtering relevant users...")
    relevant_users = set()

    # Add users who reviewed the filtered businesses
    for biz_id in business_ids:
        if biz_id in reviews:
            for review in reviews[biz_id]:
                relevant_users.add(review["user_id"])

    # Add friends if provided
    if friends:
        relevant_users.update(friends)

    # Ensure the target user is included
    if target_user_id:
        relevant_users.add(target_user_id)

    logging.info(f"Found {len(relevant_users)} relevant users.")
    return list(relevant_users)


def create_filtered_user_business_matrix(reviews, user_ids, business_ids):
    """
    Creates a sparse user-business matrix for filtered users and businesses.
    """
    logging.info("Creating user-business matrix with filtered data...")

    # Map filtered IDs to matrix indices
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    business_id_to_index = {biz_id: idx for idx, biz_id in enumerate(business_ids)}

    # Build sparse matrix data
    rows, cols, data = [], [], []
    for biz_id in business_ids:
        if biz_id in reviews:
            for review in reviews[biz_id]:
                if review["user_id"] in user_id_to_index and "stars" in review:
                    rows.append(user_id_to_index[review["user_id"]])
                    cols.append(business_id_to_index[biz_id])
                    data.append(review["stars"])

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(business_ids)))
    logging.info(f"Filtered user-business matrix created with shape {sparse_matrix.shape}.")
    return sparse_matrix


def compute_similarity(matrix, axis=0):
    """
    Computes cosine similarity along the specified axis (0 for rows, 1 for columns).
    """
    logging.info(f"Computing {'user' if axis == 0 else 'item'}-based cosine similarity...")
    if axis == 1:
        matrix = matrix.T
    similarity_matrix = cosine_similarity(matrix)
    logging.info("Cosine similarity computation complete.")
    return similarity_matrix


def recommend_businesses(user_id, user_ids, business_ids, sparse_matrix, similarity_matrix, top_k=5, axis=0):
    """
    Recommends businesses for a given user using either user-based or item-based CF.
    """
    logging.info(f"Generating {'user' if axis == 0 else 'item'}-based recommendations for user {user_id}...")

    if user_id not in user_ids:
        logging.warning(f"User ID '{user_id}' not found.")
        return []

    user_idx = user_ids.index(user_id)

    if axis == 0:  # User-based CF
        user_ratings = sparse_matrix[user_idx].toarray().flatten()
        scores = similarity_matrix[user_idx] @ sparse_matrix.toarray()
        scores /= np.abs(similarity_matrix[user_idx]).sum()
    else:  # Item-based CF
        user_ratings = sparse_matrix[user_idx].toarray().flatten()
        scores = sparse_matrix.toarray() @ similarity_matrix
        scores = scores[user_idx] / np.abs(similarity_matrix).sum(axis=0)

    # Exclude businesses the user has already rated
    rated_indices = user_ratings.nonzero()[0]
    scores[rated_indices] = -1  # Set rated businesses to a low score

    # Get top recommendations
    recommendations = [(business_ids[i], score) for i, score in enumerate(scores)]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_k]


def filter_recommendations_by_rating(recommendations, businesses, min_rating):
    """
    Filters recommendations to include only businesses with a rating >= min_rating.
    """
    logging.info(f"Filtering recommendations with minimum rating {min_rating}...")
    filtered_recommendations = []
    for biz_id, score in recommendations:
        business = next(
            (biz for city in businesses.values()
             for cat in city.values()
             for biz in cat if biz["business_id"] == biz_id), 
            None
        )
        if business and business.get("stars", 0) >= min_rating:
            filtered_recommendations.append((biz_id, score))
    logging.info(f"Filtered recommendations to {len(filtered_recommendations)} businesses.")
    return filtered_recommendations


def display_recommendations(recommendations, business_ids, businesses):
    """
    Display recommendations with business names, IDs, and scores in a formatted table.
    """
    business_name_map = get_business_names([rec[0] for rec in recommendations], businesses)
    formatted_output = []
    for biz_id, score in recommendations:
        business_name = business_name_map.get(biz_id, "Unknown Business")
        formatted_output.append((business_name, biz_id, round(score, 2)))
    return formatted_output

def get_business_names(business_ids, businesses):
    """
    Maps business IDs to their names using the business dataset.
    """
    logging.info("Mapping business IDs to names...")
    business_name_map = {}
    for city, categories in businesses.items():
        for category, biz_list in categories.items():
            for business in biz_list:
                business_name_map[business["business_id"]] = business.get("name", "Unknown Business")
    return {biz_id: business_name_map.get(biz_id, "Unknown Business") for biz_id in business_ids}


def print_recommendations(title, recommendations):
    """
    Print recommendations in a tabular format.
    """
    if not recommendations:
        print(f"{title}: No recommendations found.\n")
        return

    print(f"{title}:\n")
    print(f"{'Business Name':<40} {'Business ID':<30} {'Score':<10}")
    print("-" * 80)
    for business_name, biz_id, score in recommendations:
        print(f"{business_name:<40} {biz_id:<30} {score:<10}")
    print()


# Main Workflow
if __name__ == "__main__":
    reviews_file = "grouped_reviews.json"
    business_file = "nested_business.json"
    users_file = "ca_final_users.json"
    city = "Santa Barbara"
    query_type = "Restaurants"
    min_rating = 2.0
    example_user_id = "BqcFc5DWEPo-U6_uAY9k3Q"

    logging.info("Starting recommendation system process...")

    # Step 1: Load Data
    reviews, businesses, users = load_data(reviews_file, business_file, users_file)

    # Step 2: Filter Businesses by Criteria
    city_filtered_ids = filter_businesses_by_criteria(businesses, city, query_type, min_rating)

    # Step 3: Create User Sets
    friends = get_friends(example_user_id, users)
    valid_friends = [friend for friend in friends if any(user["user_id"] == friend for user in users)]
    valid_friends.append(example_user_id)
    logging.info(f"Valid friends of user {example_user_id}: {valid_friends}")

    # Filter relevant users
    all_reviewers = filter_relevant_users(reviews, city_filtered_ids, target_user_id=example_user_id)
    combined_users = filter_relevant_users(reviews, city_filtered_ids, friends=valid_friends, target_user_id=example_user_id)

    # Create sparse matrices
    sparse_matrix_friends = create_filtered_user_business_matrix(reviews, valid_friends, city_filtered_ids)
    sparse_matrix_all_reviewers = create_filtered_user_business_matrix(reviews, all_reviewers, city_filtered_ids)
    sparse_matrix_combined = create_filtered_user_business_matrix(reviews, combined_users, city_filtered_ids)

    # Step 5: Compute and Display Recommendations
    for matrix_name, sparse_matrix, user_ids in [
        ("Friends Only", sparse_matrix_friends, valid_friends),
        ("All Reviewers", sparse_matrix_all_reviewers, all_reviewers),
        ("Combined", sparse_matrix_combined, combined_users),
    ]:
        logging.info(f"Processing {matrix_name} Matrix...")

        # User-Based Recommendations
        user_similarity_matrix = compute_similarity(sparse_matrix, axis=0)
        user_recommendations = recommend_businesses(
            example_user_id, user_ids, city_filtered_ids, sparse_matrix, user_similarity_matrix, top_k=5, axis=0
        )
        user_recommendations = filter_recommendations_by_rating(user_recommendations, businesses, min_rating)
        formatted_user_recommendations = display_recommendations(user_recommendations, city_filtered_ids, businesses)
        print_recommendations(f"User-Based Recommendations for {matrix_name}", formatted_user_recommendations)

        # Item-Based Recommendations
        item_similarity_matrix = compute_similarity(sparse_matrix, axis=1)
        item_recommendations = recommend_businesses(
            example_user_id, user_ids, city_filtered_ids, sparse_matrix, item_similarity_matrix, top_k=5, axis=1
        )
        item_recommendations = filter_recommendations_by_rating(item_recommendations, businesses, min_rating)
        formatted_item_recommendations = display_recommendations(item_recommendations, city_filtered_ids, businesses)
        print_recommendations(f"Item-Based Recommendations for {matrix_name}", formatted_item_recommendations)
