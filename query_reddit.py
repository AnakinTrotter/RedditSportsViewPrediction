import pandas as pd
import requests
from datetime import datetime, timedelta
import urllib.parse
import os

# Constants
SCORE_THRESHOLD = 2  # Set the score threshold here
NUM_DAYS = 3  # Total number of days to query
FETCH_WINDOW_DAYS = 3  # Number of days per fetch window

# Function to fetch posts from PullPush API
def fetch_reddit_posts(name, teams, start_datetime_iso, score_threshold=SCORE_THRESHOLD, num_days=NUM_DAYS, fetch_window_days=FETCH_WINDOW_DAYS):
    try:
        # Convert start datetime to a datetime object
        start_time = datetime.fromisoformat(start_datetime_iso)
    except ValueError:
        print(f"Invalid start datetime format for {name}. Skipping.")
        return None

    # Extract everything before "Game" or "game" in the name
    if "Game" in name or "game" in name:
        base_name = name.split("Game")[0].strip()
    else:
        base_name = name.strip()

    # Build query terms
    query_terms = [f'"{base_name}"']  # Base name in quotes
    query_terms += [f'"{team.strip()}"' for team in teams.split(",")]  # Add teams in quotes
    query = " OR ".join(query_terms)
    encoded_query = urllib.parse.quote(query)  # URL encode the query string

    # Initialize counters
    total_posts = 0
    total_comments = 0
    total_scores = 0

    # Query in fetch_window_days chunks
    for day_offset in range(num_days, 0, -fetch_window_days):
        # Calculate 'since' and 'until' for each chunk
        window_end = start_time - timedelta(days=day_offset - fetch_window_days)
        window_start = start_time - timedelta(days=day_offset)

        since = int(window_start.timestamp())
        until = int(window_end.timestamp())

        # Construct the API URL with the 'score' parameter
        url = (
            f'https://api.pullpush.io/reddit/submission/search'
            f'?html_decode=True&since={since}&until={until}'
            f'&score=%3E{score_threshold}'
            f'&q={encoded_query}&size=100'
            f'&sort=desc&sort_type=score'
        )

        # Print the request URL for debugging
        print(f"Request URL: {url}")

        # Make the GET request to the API
        response = requests.get(url)

        # Check for successful response
        if response.status_code == 200:
            data = response.json().get('data', [])
            total_posts += len(data)
            total_comments += sum(post.get('num_comments', 0) for post in data)
            total_scores += sum(post.get('score', 0) for post in data)
        else:
            print(f"Error fetching data for {name} from {window_start} to {window_end}: {response.status_code}")
            continue

    return total_posts, total_comments, total_scores

# Read input file
input_file = "tv_data.csv"
output_file = "reddit_metrics.csv"

# Read the input CSV
tv_data_df = pd.read_csv(input_file)

# Check if the output CSV exists
if os.path.exists(output_file):
    reddit_metrics_df = pd.read_csv(output_file)
    processed_names = set(reddit_metrics_df["Name"].tolist())
else:
    reddit_metrics_df = pd.DataFrame(columns=[
        "Name", "Year", "Total Posts", "Total Comments",
        "Total Scores", "Avg Comments Per Post", "Avg Score Per Post"
    ])
    processed_names = set()

# Process each row in the input file
for index, row in tv_data_df.iterrows():
    name = row['Name']
    if name in processed_names:
        print(f"Skipping already processed {name}")
        continue

    teams = row['Teams']
    start_datetime_iso = row['Start Datetime']
    print(f"\nFetching posts for: {name}")

    # Fetch posts
    result = fetch_reddit_posts(name, teams, start_datetime_iso)

    if result:
        total_posts, total_comments, total_scores = result

        # Calculate averages
        avg_comments_per_post = total_comments / total_posts if total_posts > 0 else 0
        avg_score_per_post = total_scores / total_posts if total_posts > 0 else 0

        # Append the result using pd.concat
        result_row = {
            "Name": name,
            "Year": datetime.fromisoformat(start_datetime_iso).year,
            "Total Posts": total_posts,
            "Total Comments": total_comments,
            "Total Scores": total_scores,
            "Avg Comments Per Post": avg_comments_per_post,
            "Avg Score Per Post": avg_score_per_post
        }

        result_row_df = pd.DataFrame([result_row])  # Create a one-row DataFrame
        reddit_metrics_df = pd.concat([reddit_metrics_df, result_row_df], ignore_index=True)

        # Write to CSV
        reddit_metrics_df.to_csv(output_file, index=False)
        print(f"Appended results for {name} to {output_file}")

print("\nProcessing completed!")
