import requests
from datetime import datetime, timedelta
import urllib.parse
import pandas as pd
import os
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk

# Ensure Vader Lexicon is downloaded
nltk.download("vader_lexicon")

# Constants
NUM_HOURS = 168              # Total number of hours to query
FETCH_WINDOW_HOURS = 168     # Number of hours per fetch window
MAX_DATA_POINTS = 77        # Number of data points to process
SPORT_TO_PROCESS = "World_Series"  # Specify sport to process ("all" for all sports)
SUBREDDIT_MAP = {           # Map sport types to subreddits
    "World_Series": "baseball",
    "Super_Bowl": "nfl",
    "NBA_Finals": "nba",
    "MLS_Cup": "soccer"
}

# Initialize Vader Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function to determine the sport type from the event name
def determine_sport(name):
    name_lower = name.lower()
    if "super bowl" in name_lower:
        return "Super_Bowl"
    elif "mls cup" in name_lower:
        return "MLS_Cup"
    elif "nba finals" in name_lower:
        return "NBA_Finals"
    elif "world series" in name_lower:
        return "World_Series"
    else:
        return "Other"

# Function to fetch posts from PullPush API with sentiment analysis
def fetch_reddit_posts(name, teams, start_datetime_iso, subreddit, num_hours=NUM_HOURS, fetch_window_hours=FETCH_WINDOW_HOURS):
    try:
        # Convert start datetime to a datetime object
        start_time = datetime.fromisoformat(start_datetime_iso)
    except ValueError:
        print(f"Invalid start datetime format for {name}. Skipping.")
        return None

    # Build the query string for teams
    query_terms = [f'"{team.strip()}"' for team in teams.split(",")]
    query = " ".join(query_terms)
    encoded_query = urllib.parse.quote(query)

    # Initialize counters
    total_posts = 0
    total_comments = 0
    total_scores = 0
    total_sentiment_textblob = 0.0  # Sentiment from TextBlob
    total_sentiment_vader = 0.0    # Sentiment from Vader

    # Query in fetch_window_hours chunks
    for hour_offset in range(0, num_hours, fetch_window_hours):
        # Calculate 'since' and 'until' for each chunk
        window_start = start_time - timedelta(hours=hour_offset + fetch_window_hours)
        window_end = start_time - timedelta(hours=hour_offset)

        since = int(window_start.timestamp())
        until = int(window_end.timestamp())

        # Construct the API URL with the subreddit filter and sorting
        url = (
            f'https://api.pullpush.io/reddit/submission/search'
            f'?html_decode=True&since={since}&until={until}'
            f'&q={encoded_query}&size=100'
            f'&sort=desc&sort_type=score'
            f'&subreddit={subreddit}'
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

            # Sentiment analysis
            for post in data:
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                text = f"{title} {selftext}"
                # Sentiment from TextBlob
                total_sentiment_textblob += TextBlob(text).sentiment.polarity
                # Sentiment from Vader
                total_sentiment_vader += vader_analyzer.polarity_scores(text)['compound']
        else:
            print(f"Error fetching data for {name} from {window_start} to {window_end}: {response.status_code}")
            continue

    return total_posts, total_comments, total_scores, total_sentiment_textblob, total_sentiment_vader

# Input file
input_file = "tv_data.csv"

# Read input CSV file
data_points = pd.read_csv(input_file).to_dict("records")

# Process only the first N data points
data_points = data_points[:MAX_DATA_POINTS]

# Process each data point
for row in data_points:
    name = row["Name"]
    teams = row["Teams"]
    start_datetime_iso = row["Start Datetime"]
    viewers = row["Average Viewers (Millions)"]

    sport = determine_sport(name)
    if SPORT_TO_PROCESS != "all" and sport != SPORT_TO_PROCESS:
        continue  # Skip if the sport does not match the specified sport

    subreddit = SUBREDDIT_MAP.get(sport, None)
    if not subreddit:
        print(f"No subreddit found for sport: {sport}. Skipping.")
        continue

    print(f"\nFetching posts for: {name} ({sport}) in subreddit: {subreddit}")

    # Fetch posts
    result = fetch_reddit_posts(name, teams, start_datetime_iso, subreddit)

    if result:
        # Combine results
        total_posts, total_comments, total_scores, total_sentiment_textblob, total_sentiment_vader = result

        # Calculate average sentiments
        avg_sentiment_textblob = total_sentiment_textblob / total_posts if total_posts > 0 else 0
        avg_sentiment_vader = total_sentiment_vader / total_posts if total_posts > 0 else 0

        # Append the result
        result_row = {
            "Name": name,
            "Year": datetime.fromisoformat(start_datetime_iso).year,
            "Teams": teams,
            "Total Posts": total_posts,
            "Total Comments": total_comments,
            "Total Scores": total_scores,
            "Avg Sentiment (TextBlob)": avg_sentiment_textblob,
            "Avg Sentiment (Vader)": avg_sentiment_vader,
            "Viewers (Millions)": viewers
        }

        # Write to the CSV file for the sport
        output_file = f"{sport}.csv"
        if os.path.exists(output_file):
            # Append to existing CSV
            existing_df = pd.read_csv(output_file)
            updated_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
            updated_df.to_csv(output_file, index=False)
        else:
            # Create a new CSV
            pd.DataFrame([result_row]).to_csv(output_file, index=False)
        print(f"Updated {output_file} with data for {name}")
    else:
        print(f"No data retrieved for {name}.")

# After processing, plot all features vs. viewership
if SPORT_TO_PROCESS != "all":
    output_file = f"{SPORT_TO_PROCESS}.csv"
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)

        # Plot all features against viewership
        features = [
            "Total Posts", "Total Comments", "Total Scores",
            "Avg Sentiment (TextBlob)", "Avg Sentiment (Vader)"
        ]

        for feature in features:
            if feature in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df[feature], df['Viewers (Millions)'], label=feature, alpha=0.7)
                plt.title(f'{feature} vs. Viewership')
                plt.xlabel(feature)
                plt.ylabel('Viewers (Millions)')
                plt.legend()
                plt.grid(True)
                plt.show()
    else:
        print(f"No data available in {output_file} to plot.")
