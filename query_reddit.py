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
NUM_HOURS = 72              # Total number of hours to query
MAX_WINDOW_HOURS = 36       # Maximum number of hours per fetch window
MAX_DATA_POINTS = None      # Number of data points to process (None = no max)
SPORT_TO_PROCESS = "all"  # Specify sport to process ("all" for all sports)
SUBREDDIT_MAP = {           # Map sport types to subreddits
    "World_Series": "baseball",
    "Super_Bowl": "nfl",
    "NBA_Finals": "nba",
    "MLS_Cup": "soccer",
    "Stanley_Cup": "nhl"
}

EVENT_KEYWORDS = {          # Map sport types to event keywords
    "World_Series": "World Series",
    "Super_Bowl": "Super Bowl",
    "NBA_Finals": "NBA Finals",
    "MLS_Cup": "MLS Cup",
    "Stanley_Cup": "Stanley Cup"
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
    elif "stanley cup" in name_lower:
        return "Stanley_Cup"
    else:
        return "Other"

# Function to fetch posts with dynamic window adjustment
def fetch_reddit_posts_dynamic(name, teams, start_datetime_iso, sport, subreddit, num_hours=NUM_HOURS, max_window_hours=MAX_WINDOW_HOURS):
    try:
        # Convert start datetime to a datetime object
        start_time = datetime.fromisoformat(start_datetime_iso)
    except ValueError:
        print(f"Invalid start datetime format for {name}. Skipping.")
        return None

    # Build the query string with event and team names
    event_keyword = EVENT_KEYWORDS.get(sport, "")
    query_terms = [f'"{event_keyword}"'] + [f'"{team.strip()}"' for team in teams.split(",")]
    query = " OR ".join(query_terms)
    encoded_query = urllib.parse.quote(query)

    # Initialize counters
    total_posts, total_comments, total_scores = 0, 0, 0
    total_sentiment_textblob, total_sentiment_vader = 0.0, 0.0

    # Query in dynamically adjusted windows
    remaining_hours = num_hours
    current_window_hours = max_window_hours

    while remaining_hours > 0:
        # Adjust the current window size
        current_window_hours = min(current_window_hours, remaining_hours)

        # Calculate time range
        window_start = start_time - timedelta(hours=remaining_hours)
        window_end = window_start + timedelta(hours=current_window_hours)

        since, until = int(window_start.timestamp()), int(window_end.timestamp())

        # API URL
        url = (
            f'https://api.pullpush.io/reddit/submission/search'
            f'?html_decode=True&since={since}&until={until}'
            f'&q={encoded_query}&size=100'
            f'&sort=desc&sort_type=score'
            f'&subreddit={subreddit}'
        )
        print(f"Request URL: {url}")

        # Make API request
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('data', [])
            items_count = len(data)

            # Adjust window size
            if items_count == 100:
                current_window_hours = max(1, current_window_hours // 2)
                print(f"Hit 100-item limit. Reducing window to {current_window_hours} hours.")
                continue
            elif items_count < 100 and current_window_hours < max_window_hours:
                current_window_hours = min(max_window_hours, current_window_hours * 2)

            # Process data
            total_posts += items_count
            total_comments += sum(post.get('num_comments', 0) for post in data)
            total_scores += sum(post.get('score', 0) for post in data)

            for post in data:
                text = f"{post.get('title', '')} {post.get('selftext', '')}"
                total_sentiment_textblob += TextBlob(text).sentiment.polarity
                total_sentiment_vader += vader_analyzer.polarity_scores(text)['compound']

            remaining_hours -= current_window_hours
        else:
            print(f"Error fetching data: {response.status_code}")
            break

    return total_posts, total_comments, total_scores, total_sentiment_textblob, total_sentiment_vader

# Input file
input_file = "tv_data.csv"

# Read input CSV
data_points = pd.read_csv(input_file).to_dict("records")
if MAX_DATA_POINTS:
    data_points = data_points[:MAX_DATA_POINTS]

# Process each data point
results = []
for row in data_points:
    name, teams, start_datetime_iso, viewers = row["Name"], row["Teams"], row["Start Datetime"], row["Average Viewers (Millions)"]

    sport = determine_sport(name)
    if SPORT_TO_PROCESS != "all" and sport != SPORT_TO_PROCESS:
        continue

    subreddit = SUBREDDIT_MAP.get(sport)
    if not subreddit:
        print(f"No subreddit found for sport: {sport}. Skipping.")
        continue

    print(f"\nFetching posts for: {name} ({sport}) in subreddit: {subreddit}")
    result = fetch_reddit_posts_dynamic(name, teams, start_datetime_iso, sport, subreddit)

    if result:
        total_posts, total_comments, total_scores, total_sentiment_textblob, total_sentiment_vader = result

        if total_posts == 0:
            print(f"Skipping {name} due to 0 posts.")
            continue

        avg_sentiment_textblob = total_sentiment_textblob / total_posts if total_posts > 0 else 0
        avg_sentiment_vader = total_sentiment_vader / total_posts if total_posts > 0 else 0

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
        results.append(result_row)

        # Write to file
        output_file = f"{sport}.csv"
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True).to_csv(output_file, index=False)
        else:
            pd.DataFrame([result_row]).to_csv(output_file, index=False)
        print(f"Updated {output_file} with data for {name}")
    else:
        print(f"No data retrieved for {name}.")
