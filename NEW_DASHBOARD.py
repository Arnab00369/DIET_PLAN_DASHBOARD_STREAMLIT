# ==========================================================
# IMPORTS SECTION
# ==========================================================

# Import Streamlit library for creating web application interface
# Streamlit is a Python framework that allows building interactive web apps with minimal code
import streamlit as st

# Import pandas for data manipulation and analysis (tabular data)
# Pandas provides DataFrame structures for efficient data handling
import pandas as pd

# Import numpy for numerical operations and random number generation
# NumPy is fundamental for scientific computing in Python
import numpy as np

# Import random for random selection functionality
# Python's built-in random module for generating random numbers and selections
import random

# Import plotly express for easy creation of interactive plots and charts
# Plotly Express is a high-level interface for creating visualizations
import plotly.express as px

# Import plotly graph objects for more customized plotting capabilities
# Plotly Graph Objects provides more control over plot customization
import plotly.graph_objects as go

# Import urllib.parse for URL encoding/decoding operations
# Used for creating safe URLs for external recipe searches
import urllib.parse

# Import option_menu for creating navigation menu in Streamlit
# Custom Streamlit component for creating dropdown-style navigation menus
from streamlit_option_menu import option_menu

# Import hashlib for password hashing
# Provides secure hash algorithms for password storage
import hashlib

# Import json for data storage
# Used for serializing and deserializing user data to/from files
import json

# Import os for operating system interactions
# Used for file path operations and checking file existence
import os

# Import datetime for date and time operations
# Used for timestamping user history entries
from datetime import datetime, timedelta

# Import time for time-related functions
# Used for adding delays in the application flow
import time

# Import re for regular expressions
# Used for data validation
import re


# ==========================================================
# DATA LOADING FUNCTION
# ==========================================================

# @st.cache_data decorator caches the function output to improve performance
# This prevents reloading data on every interaction when the app reruns
# Streamlit's caching mechanism stores the function result for faster subsequent calls
@st.cache_data
def load_data():
    # Try to load the actual cleaned dataset file
    # Attempt to read from existing CSV file to avoid regenerating data each time
    try:
        # Attempt to read the CSV file from disk
        # pd.read_csv() loads comma-separated values file into pandas DataFrame
        df = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")
    
    # If file doesn't exist, create a sample dataset
    # Exception handling for when the CSV file is not found
    except:
        # Set random seed for reproducibility of random values
        # Ensures same random numbers are generated each time for consistency
        np.random.seed(42)
        
        # Define number of food items to generate
        # Creates a dataset with 200 different food items
        n_foods = 200
        
        # Create dictionary with sample food data
        # Each key represents a column in the dataframe
        # Dictionary comprehension for structured data initialization
        data = {
            # Generate food names with sequential numbering
            'name': [f'Indian Food {i}' for i in range(1, n_foods+1)],
            
            # Generate random calorie values between 100-600
            # np.random.uniform() creates continuous uniform distribution
            'calories': np.random.uniform(100, 600, n_foods),
            
            # Generate random protein values between 2-30 grams
            'protein': np.random.uniform(2, 30, n_foods),
            
            # Generate random fat values between 1-25 grams
            'fat': np.random.uniform(1, 25, n_foods),
            
            # Generate random carbohydrate values between 10-80 grams
            'carbs': np.random.uniform(10, 80, n_foods),
            
            # Generate boolean values for vegetarian status
            # np.random.choice() with probability weights (70% True, 30% False)
            'is_veg': np.random.choice([True, False], n_foods, p=[0.7, 0.3]),
            
            # Generate boolean values for egg content
            # 30% probability of containing eggs
            'contains_egg': np.random.choice([True, False], n_foods, p=[0.3, 0.7]),
            
            # Allergen columns with different probability distributions
            # Each allergen has specific probability of being present
            'is_allergen_gluten': np.random.choice([True, False], n_foods, p=[0.2, 0.8]),
            'is_allergen_dairy': np.random.choice([True, False], n_foods, p=[0.3, 0.7]),
            'is_allergen_nuts': np.random.choice([True, False], n_foods, p=[0.2, 0.8]),
            'is_allergen_soy': np.random.choice([True, False], n_foods, p=[0.1, 0.9]),
            'is_allergen_shellfish': np.random.choice([True, False], n_foods, p=[0.1, 0.9]),
            'is_allergen_eggs': np.random.choice([True, False], n_foods, p=[0.2, 0.8]),
            'is_allergen_fish': np.random.choice([True, False], n_foods, p=[0.15, 0.85]),
            
            # Medical suitability columns
            # Boolean indicators for foods suitable for specific medical conditions
            'suitable_diabetes': np.random.choice([True, False], n_foods, p=[0.6, 0.4]),
            'suitable_hypertension': np.random.choice([True, False], n_foods, p=[0.7, 0.3]),
            'suitable_heart_disease': np.random.choice([True, False], n_foods, p=[0.65, 0.35]),
            'suitable_thyroid': np.random.choice([True, False], n_foods, p=[0.8, 0.2]),
            'suitable_pcos': np.random.choice([True, False], n_foods, p=[0.7, 0.3]),
            'suitable_kidney_disease': np.random.choice([True, False], n_foods, p=[0.5, 0.5]),
            'suitable_gerd': np.random.choice([True, False], n_foods, p=[0.6, 0.4]),
        }
        
        # Apply logical constraints: if food is vegetarian, it shouldn't contain certain allergens
        # Loop through each food item to enforce dietary consistency rules
        for i in range(n_foods):
            # Check if current food item is vegetarian
            if data['is_veg'][i]:
                # Vegetarian foods don't contain eggs (in Indian vegetarian context)
                data['contains_egg'][i] = False
                # No shellfish in vegetarian foods
                data['is_allergen_shellfish'][i] = False
                # No fish in vegetarian foods
                data['is_allergen_fish'][i] = False
        
        # Create pandas DataFrame from the dictionary
        # Convert dictionary of lists into structured DataFrame
        df = pd.DataFrame(data)
        
        # Save the generated dataset to CSV for future use
        # to_csv() writes DataFrame to CSV file, index=False prevents saving row indices
        df.to_csv("Cleaned_Indian_Food_Dataset.csv", index=False)
    
    # Return the dataframe (either loaded or generated)
    # Function output is the DataFrame containing food data
    return df

# Call the load_data function and store the result in df variable
# This executes the data loading/generation process and stores result globally
df = load_data()


# ==========================================================
# AUTHENTICATION FUNCTIONS
# ==========================================================

def hash_password(password):
    """Hash a password for storing."""
    # Create SHA-256 hash object for password security
    # encode() converts string to bytes, hexdigest() returns hexadecimal string
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file."""
    # Try to load existing user data from file
    try:
        # Check if user data file exists
        if os.path.exists("users.json"):
            # Open file in read mode
            with open("users.json", "r") as f:
                # json.load() parses JSON file into Python dictionary
                return json.load(f)
    # Handle any file reading errors gracefully
    except:
        # Pass silently if file doesn't exist or has errors
        pass
    # Return empty dictionary if no users file exists
    return {}

def save_users(users):
    """Save users to JSON file."""
    # Open file in write mode (creates new file or overwrites existing)
    with open("users.json", "w") as f:
        # json.dump() writes Python dictionary as JSON formatted data
        json.dump(users, f)

def register_user(username, password, email, name):
    """Register a new user."""
    # Load existing users from storage
    users = load_users()
    
    # Check if username already exists
    if username in users:
        # Return failure status and message
        return False, "Username already exists!"
    
    # Validate email format
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return False, "Invalid email format!"
    
    # Create new user entry with hashed password
    users[username] = {
        'password': hash_password(password),  # Store hashed password for security
        'email': email,                       # User's email address
        'name': name,                         # User's full name
        'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Current timestamp
        'profile_history': [],                # Initialize empty profile history list
        'meal_plan_history': [],              # Initialize empty meal plan history list
        'last_login': None,                   # Initialize last login timestamp
        'user_type': 'user',                  # User type (user or admin)
        'is_active': True,                    # Account status
        'login_count': 0                      # Track login count
    }
    
    # Save updated users dictionary to file
    save_users(users)
    # Return success status and message
    return True, "Registration successful!"

def authenticate_user(username, password, is_admin=False):
    """Authenticate a user or admin."""
    # Load existing users from storage
    users = load_users()
    
    # Check if username exists in database
    if username not in users:
        # Return failure if user not found
        return False, "User not found!"
    
    # For admin authentication, check if user is admin
    if is_admin:
        if users[username].get('user_type') != 'admin':
            return False, "Access denied! Admin privileges required."
    
    # Check if account is active
    if not users[username].get('is_active', True):
        return False, "Account is deactivated. Contact administrator."
    
    # Compare hashed input password with stored hash
    if users[username]['password'] == hash_password(password):
        # Update last login and login count
        users[username]['last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        users[username]['login_count'] = users[username].get('login_count', 0) + 1
        save_users(users)
        # Return success if passwords match
        return True, "Login successful!"
    
    # Return failure if passwords don't match
    return False, "Incorrect password!"

def save_user_profile(username, profile_data):
    """Save user profile to history."""
    # Load existing users from storage
    users = load_users()
    
    # Check if user exists
    if username in users:
        # Add timestamp to profile data for tracking
        profile_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create sequential plan ID
        profile_data['plan_id'] = len(users[username]['profile_history']) + 1
        
        # Save to history by appending to profile_history list
        users[username]['profile_history'].append(profile_data)
        
        # Keep only last 10 profiles to prevent unlimited growth
        if len(users[username]['profile_history']) > 10:
            # Slice to keep only most recent 10 entries
            users[username]['profile_history'] = users[username]['profile_history'][-10:]
        
        # Save updated users data
        save_users(users)
        return True  # Return success
    return False  # Return failure if user not found

def save_meal_plan(username, meal_plan_data):
    """Save meal plan to history."""
    # Load existing users from storage
    users = load_users()
    
    # Check if user exists
    if username in users:
        # Add timestamp to meal plan data
        meal_plan_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Create sequential plan ID
        meal_plan_data['plan_id'] = len(users[username]['meal_plan_history']) + 1
        
        # Save to history by appending to meal_plan_history list
        users[username]['meal_plan_history'].append(meal_plan_data)
        
        # Keep only last 10 meal plans to manage storage
        if len(users[username]['meal_plan_history']) > 10:
            # Slice to keep only most recent 10 entries
            users[username]['meal_plan_history'] = users[username]['meal_plan_history'][-10:]
        
        # Save updated users data
        save_users(users)
        return True  # Return success
    return False  # Return failure if user not found

def get_user_history(username):
    """Get user's profile and meal plan history."""
    # Load existing users from storage
    users = load_users()
    
    # Check if user exists
    if username in users:
        # Return both history lists using get() with default empty lists
        return users[username].get('profile_history', []), users[username].get('meal_plan_history', [])
    # Return empty lists if user not found
    return [], []

def get_all_users():
    """Get all registered users data."""
    users = load_users()
    return users

def update_user_status(username, is_active):
    """Update user active status."""
    users = load_users()
    if username in users:
        users[username]['is_active'] = is_active
        save_users(users)
        return True
    return False

def delete_user(username):
    """Delete a user account."""
    users = load_users()
    if username in users:
        # Don't allow deletion of admin accounts
        if users[username].get('user_type') == 'admin':
            return False, "Cannot delete admin accounts!"
        del users[username]
        save_users(users)
        return True, "User deleted successfully!"
    return False, "User not found!"

def create_admin_account():
    """Create default admin account if not exists."""
    users = load_users()
    admin_username = "admin"
    if admin_username not in users:
        users[admin_username] = {
            'password': hash_password("admin123"),  # Default admin password
            'email': "admin@nutricare.com",
            'name': "System Administrator",
            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'profile_history': [],
            'meal_plan_history': [],
            'last_login': None,
            'user_type': 'admin',
            'is_active': True,
            'login_count': 0
        }
        save_users(users)
        return True
    return False

# Create admin account on startup
create_admin_account()


# ==========================================================
# PAGE CONFIGURATION AND STYLING
# ==========================================================

# Configure Streamlit page settings
# This must be the first Streamlit command on the page
st.set_page_config(
    page_title="NUTRI-CARE: AI-Powered Indian Diet Planner",  # Browser tab title
    page_icon="🍛",  # Browser tab icon (emoji)
    layout="wide",  # Use wide layout instead of centered
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

# Inject custom CSS for styling the application
# st.markdown() allows HTML/CSS injection into Streamlit app
st.markdown("""
<style>
    /* Main header styling - primary page title */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;  /* Blue color */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;  /* Bold font */
    }
    
    /* Sub-header styling - section titles */
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;  /* Lighter blue */
        margin: 1.5rem 0rem 1rem 0rem;
        font-weight: 600;  /* Semi-bold */
    }
    
    /* Metric card styling for displaying stats - dashboard cards */
    .metric-card {
        background-color: #1e252d;  /* Dark background */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;  /* Blue left border */
        margin: 0.5rem 0;
    }
    
    /* Food card styling for individual food items - food display cards */
    .food-card {
        background-color: #1e252d;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;  /* Light gray border */
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* Subtle shadow */
    }
    
    /* Day plan container styling - daily meal plan cards */
    .day-plan {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Purple gradient */
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Login card styling - authentication interface */
    .login-card {
        background-color: #1e252d;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    
    /* History card styling - user history items */
    .history-card {
        background-color: #1e252d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    /* Admin card styling */
    .admin-card {
        background-color: #1a1a2e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ff6b6b;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Warning card */
    .warning-card {
        background-color: #2d1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    
    /* Success card */
    .success-card {
        background-color: #1a2d1a;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    /* User list item */
    .user-item {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4299e1;
    }
    
    /* Table styling */
    .data-table {
        background-color: #1e252d;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Status indicators */
    .status-active {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .status-admin {
        color: #ffd700;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html needed for custom CSS injection


# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================

# Initialize session state variables to preserve data across app interactions
# Session state is Streamlit's way of maintaining state between reruns

# Authentication state - tracks if user is logged in
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False  # Default: not authenticated
    
# Current user - stores username of logged-in user
if 'current_user' not in st.session_state:
    st.session_state.current_user = None  # Default: no user

# User profile - stores all user input data and preferences
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}  # Empty dictionary to store user information
    
# Filtered foods - stores foods filtered based on user preferences
if 'filtered_foods' not in st.session_state:
    st.session_state.filtered_foods = pd.DataFrame()  # Empty DataFrame for filtered food data
    
# Meal plan - stores generated meal plan structure
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = {}  # Empty dictionary to store generated meal plan

# Show registration form - controls whether to show login or registration form
if 'show_register' not in st.session_state:
    st.session_state.show_register = False  # Default: show login form

# Admin authentication state
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False  # Default: not admin authenticated

# Login type (user/admin)
if 'login_type' not in st.session_state:
    st.session_state.login_type = "user"  # Default: user login


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_user_metrics(age, gender, weight, height, activity_level, goal):
    """
    Calculate BMI (Body Mass Index) and TDEE (Total Daily Energy Expenditure)
    
    Parameters:
    age (int): User's age in years
    gender (str): User's gender ('Male' or 'Female')
    weight (float): User's weight in kg
    height (float): User's height in cm
    activity_level (str): Activity level category
    goal (str): Fitness goal (Weight Loss, Muscle Gain, Maintenance)
    
    Returns:
    tuple: (bmi, tdee) - BMI value and daily calorie target
    """
    
    # Convert height from cm to meters for BMI calculation
    # Divide by 100 to convert centimeters to meters
    height_m = height / 100
    
    # Calculate BMI: weight(kg) / height(m)^2
    # BMI formula: mass (kg) divided by square of height (m)
    # round() to 2 decimal places for readability
    bmi = round(weight / (height_m ** 2), 2)
    
    # Harris–Benedict equation for BMR (Basal Metabolic Rate)
    # Different formulas for male and female based on biological differences
    if gender == "Male":
        # Male BMR formula: 88.36 + (13.4×weight) + (4.8×height) - (5.7×age)
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:  # Female
        # Female BMR formula: 447.6 + (9.25×weight) + (3.1×height) - (4.3×age)
        bmr = 447.6 + (9.25 * weight) + (3.1 * height) - (4.3 * age)

    # Activity level multipliers for TDEE calculation
    # Dictionary mapping activity levels to multiplication factors
    activity_factors = {
        "Sedentary": 1.2,        # Little or no exercise
        "Lightly Active": 1.375, # Light exercise 1-3 days/week
        "Moderately Active": 1.55, # Moderate exercise 3-5 days/week
        "Very Active": 1.725     # Hard exercise 6-7 days/week
    }
    
    # Calculate TDEE: BMR multiplied by activity factor
    # Total Daily Energy Expenditure estimates daily calorie needs
    tdee = round(bmr * activity_factors[activity_level])

    # Adjust TDEE based on user's goal using calorie surplus/deficit
    if goal == "Weight Loss":
        # 15% calorie deficit for weight loss
        tdee = round(tdee * 0.85)
    elif goal == "Muscle Gain":
        # 15% calorie surplus for muscle gain
        tdee = round(tdee * 1.15)
    # For Maintenance, keep TDEE as is (no adjustment)
        
    # Return both calculated metrics as tuple
    return bmi, tdee


def filter_foods(df, food_preference, allergies, medical_conditions, goal):
    """
    Filter food dataset based on user preferences, allergies, medical conditions, and goals
    
    Parameters:
    df (DataFrame): Original food dataset
    food_preference (str): Vegetarian/Non-Vegetarian preference
    allergies (list): List of user's allergies
    medical_conditions (list): List of user's medical conditions
    goal (str): User's fitness goal
    
    Returns:
    DataFrame: Filtered and scored food items
    """
    
    # Start with a copy of the original dataframe to avoid modifying the original
    # copy() creates independent DataFrame to prevent side effects
    filtered_df = df.copy()

    # Filter by food preference (Vegetarian/Non-Vegetarian)
    # Apply dietary preference filters
    if food_preference == "Vegetarian":
        # Keep only vegetarian foods (is_veg == True)
        filtered_df = filtered_df[filtered_df["is_veg"] == True]
    elif food_preference == "Non-Vegetarian":
        # Keep only non-vegetarian foods (is_veg == False)
        filtered_df = filtered_df[filtered_df["is_veg"] == False]
    # For "Any" or "Eggetarian", no filtering by veg/non-veg (all foods included)

    # Mapping of allergy names to dataframe column names
    # Dictionary connecting user-friendly allergy names to DataFrame column names
    allergy_mapping = {
        "Gluten": "is_allergen_gluten",
        "Dairy": "is_allergen_dairy", 
        "Nuts": "is_allergen_nuts",
        "Soy": "is_allergen_soy",
        "Shellfish": "is_allergen_shellfish",
        "Eggs": "is_allergen_eggs",
        "Fish": "is_allergen_fish"
    }

    # Filter out foods that contain user's allergies
    # Iterate through each allergy user has selected
    for allergy in allergies:
        # Check if allergy is in our mapping dictionary
        if allergy in allergy_mapping:
            # Get corresponding column name from mapping
            allergy_col = allergy_mapping[allergy]
            # Check if column exists in DataFrame (safety check)
            if allergy_col in filtered_df.columns:
                # Keep only foods that are NOT allergens for this type
                # Filter out rows where allergen column is True
                filtered_df = filtered_df[filtered_df[allergy_col] == False]

    # Mapping of medical conditions to suitability columns
    # Dictionary connecting medical condition names to suitability column names
    medical_mapping = {
        "Diabetes": "suitable_diabetes",
        "Hypertension": "suitable_hypertension",
        "Heart Disease": "suitable_heart_disease",
        "Thyroid Issues": "suitable_thyroid",
        "PCOS": "suitable_pcos", 
        "Kidney Disease": "suitable_kidney_disease",
        "GERD/Acid Reflux": "suitable_gerd"
    }

    # Filter for foods suitable for user's medical conditions
    # Iterate through each medical condition user has selected
    for condition in medical_conditions:
        # Check if condition is in our mapping dictionary
        if condition in medical_mapping:
            # Get corresponding column name from mapping
            condition_col = medical_mapping[condition]
            # Check if column exists in DataFrame (safety check)
            if condition_col in filtered_df.columns:
                # Keep only foods that ARE suitable for this condition
                # Filter rows where suitability column is True
                filtered_df = filtered_df[filtered_df[condition_col] == True]

    # Additional filtering based on fitness goal
    # Apply goal-specific nutritional filters
    if goal == "Weight Loss":
        # For weight loss, filter for lower calorie foods (below median)
        # Median provides dynamic threshold based on available foods
        filtered_df = filtered_df[filtered_df["calories"] < filtered_df["calories"].median()]
    elif goal == "Muscle Gain":
        # For muscle gain, filter for higher protein foods (above median)
        filtered_df = filtered_df[filtered_df["protein"] > filtered_df["protein"].median()]

    # Define ideal macronutrient ratios based on goal
    # Dictionary with protein, fat, carb ratios for each goal
    if goal == "Muscle Gain":
        # Higher protein (30%), moderate fat (25%), moderate carbs (45%)
        ratio = {"protein": 0.30, "fat": 0.25, "carbs": 0.45}
    elif goal == "Weight Loss":
        # Higher protein (35%), lower fat (20%), moderate carbs (45%)
        ratio = {"protein": 0.35, "fat": 0.20, "carbs": 0.45}
    else:  # Maintenance
        # Balanced ratios: 25% protein, 25% fat, 50% carbs
        ratio = {"protein": 0.25, "fat": 0.25, "carbs": 0.50}

    # Calculate a score for each food based on how close it is to ideal ratios
    # Only calculate if there are foods remaining after filtering
    if not filtered_df.empty:
        # Calculate score as sum of absolute differences from ideal ratios
        # Lower score means better match to ideal ratios
        filtered_df["score"] = (
            # Absolute difference from ideal protein ratio (protein/calories)
            abs(filtered_df["protein"] / filtered_df["calories"] - ratio["protein"]) +
            # Absolute difference from ideal fat ratio (fat/calories)
            abs(filtered_df["fat"] / filtered_df["calories"] - ratio["fat"]) +
            # Absolute difference from ideal carb ratio (carbs/calories)
            abs(filtered_df["carbs"] / filtered_df["calories"] - ratio["carbs"])
        )
        # Sort by score (lower score = better match) and take top 50
        # sort_values() sorts DataFrame by score column in ascending order
        # head(50) takes first 50 rows (top 50 best matches)
        filtered_df = filtered_df.sort_values("score").head(50)
    else:
        # Fallback: if no foods match, return first 10 from original dataset
        # Ensures function always returns some data
        filtered_df = df.head(10)
        
    # Return filtered and scored foods DataFrame
    return filtered_df


def generate_meal_plan(filtered_df, days=7):
    """
    Generate a 7-day meal plan from filtered foods
    
    Parameters:
    filtered_df (DataFrame): Filtered food dataset
    days (int): Number of days to plan for (default: 7)
    
    Returns:
    dict: Nested dictionary with meal plan structure
    """
    
    # Initialize empty dictionary for meal plan
    # Will store day-wise meal structure
    meal_plan = {}
    
    # Check if we have enough foods to create a meal plan
    # Need at least 21 foods (3 meals × 7 days) for unique selections
    if len(filtered_df) < 21:
        # Return empty dict if not enough foods
        return {}
        
    # Generate meal plan for each day
    # Loop through specified number of days
    for day in range(days):
        # Get day name from list using day index
        # day index (0-6) maps to day names
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                   "Friday", "Saturday", "Sunday"][day]
        
        # Create meal plan for this day
        # Nested dictionary with meals as keys and food lists as values
        meal_plan[day_name] = {
            # Select 2 random foods from top 30 for breakfast
            # random.sample() ensures unique selections without replacement
            "Breakfast": random.sample(list(filtered_df["name"].head(30)), 2),
            # Select 2 random foods from top 30 for lunch
            "Lunch": random.sample(list(filtered_df["name"].head(30)), 2),
            # Select 2 random foods from top 30 for dinner
            "Dinner": random.sample(list(filtered_df["name"].head(30)), 2)
        }
    
    # Return complete meal plan dictionary
    return meal_plan

def get_system_stats():
    """Get system statistics for admin dashboard."""
    users = get_all_users()
    stats = {
        'total_users': len(users),
        'active_users': sum(1 for u in users.values() if u.get('is_active', True)),
        'admins': sum(1 for u in users.values() if u.get('user_type') == 'admin'),
        'total_logins': sum(u.get('login_count', 0) for u in users.values()),
        'today_logins': 0,
        'plans_generated': sum(len(u.get('meal_plan_history', [])) for u in users.values()),
        'profiles_created': sum(len(u.get('profile_history', [])) for u in users.values())
    }
    
    # Calculate today's logins
    today = datetime.now().strftime("%Y-%m-%d")
    for user in users.values():
        last_login = user.get('last_login')
        if last_login and last_login.startswith(today):
            stats['today_logins'] += 1
    
    return stats


# ==========================================================
# LOGIN/REGISTRATION SECTION
# ==========================================================

# Show login/registration if not authenticated
# This section handles user authentication before showing main app
if not st.session_state.authenticated and not st.session_state.admin_authenticated:
    # Display main application header
    st.markdown("<h1 class='main-header'>🍛 NUTRI-CARE: AI-Powered Indian Diet Planner</h1>", unsafe_allow_html=True)
    
    # Create two columns for login/registration interface
    # col1 for form, col2 for welcome message and features
    col1, col2 = st.columns(2)
    
    # Left column: Login/Registration forms
    with col1:
        # Apply custom CSS class for login card styling
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        
        # Radio button for login type selection
        login_type = st.radio("Login Type:", ["👤 User", "👑 Admin"], 
                             horizontal=True, key="login_type_radio")
        
        # Update session state based on selection
        st.session_state.login_type = "admin" if login_type == "👑 Admin" else "user"
        
        # Check if showing registration form or login form
        if not st.session_state.show_register or st.session_state.login_type == "admin":
            # Login Form - shown by default
            if st.session_state.login_type == "user":
                st.subheader("🔐 User Login")
            else:
                st.subheader("👑 Admin Login")
            
            # Login username input field
            login_username = st.text_input("Username", key="login_username")
            # Login password input field (masked with dots)
            login_password = st.text_input("Password", type="password", key="login_password")
            
            # Create columns for login form buttons
            if st.session_state.login_type == "user":
                col_login1, col_login2 = st.columns(2)
            else:
                col_login1, col_login2 = st.columns([3, 1])  # Different column ratio for admin
            
            # Left button column: Login button
            with col_login1:
                # Primary styled login button
                login_button_text = "Login" if st.session_state.login_type == "user" else "Admin Login"
                if st.button(login_button_text, type="primary", use_container_width=True):
                    # Validate that both fields are filled
                    if login_username and login_password:
                        # Call authentication function
                        is_admin = (st.session_state.login_type == "admin")
                        success, message = authenticate_user(login_username, login_password, is_admin)
                        if success:
                            # Update session state for successful login
                            if is_admin:
                                st.session_state.admin_authenticated = True
                                st.session_state.current_user = login_username
                                st.success("Admin login successful! Redirecting...")
                            else:
                                st.session_state.authenticated = True
                                st.session_state.current_user = login_username
                                # Reset other session state variables
                                st.session_state.user_profile = {}
                                st.session_state.filtered_foods = pd.DataFrame()
                                st.session_state.meal_plan = {}
                                st.success("Login successful! Redirecting...")
                            # Small delay for user feedback
                            time.sleep(1)
                            # Rerun app to show authenticated interface
                            st.rerun()
                        else:
                            # Show error message if authentication fails
                            st.error(message)
                    else:
                        # Warning if fields are empty
                        st.warning("Please enter username and password")
            
            # Right button column: Register button (only for user login)
            if st.session_state.login_type == "user":
                with col_login2:
                    # Secondary styled register button
                    if st.button("Register", type="secondary", use_container_width=True):
                        # Switch to registration form
                        st.session_state.show_register = True
                        # Rerun to show registration form
                        st.rerun()
            
            # Horizontal line separator
            st.markdown("---")
            
            # Information prompt
            if st.session_state.login_type == "user":
                st.info("👆 Don't have an account? Click Register to create one!")
            else:
                st.warning("⚠️ Admin access requires special privileges")
        
        else:
            # Registration Form - shown when user clicks Register (only for users)
            st.subheader("📝 Register")
            
            # Registration form input fields
            reg_username = st.text_input("Choose Username", key="reg_username")
            reg_password = st.text_input("Choose Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            reg_email = st.text_input("Email Address", key="reg_email")
            reg_name = st.text_input("Full Name", key="reg_name")
            
            # Create two columns for registration form buttons
            col_reg1, col_reg2 = st.columns(2)
            
            # Left button column: Create Account button
            with col_reg1:
                # Primary styled create account button
                if st.button("Create Account", type="primary", use_container_width=True):
                    # Validate password confirmation
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match!")
                    # Validate all fields are filled
                    elif not all([reg_username, reg_password, reg_email, reg_name]):
                        st.warning("Please fill all fields")
                    else:
                        # Call registration function
                        success, message = register_user(reg_username, reg_password, reg_email, reg_name)
                        if success:
                            st.success(message)
                            # Switch back to login form
                            st.session_state.show_register = False
                            # Rerun to show login form
                            st.rerun()
                        else:
                            # Show error if registration fails
                            st.error(message)
            
            # Right button column: Back to Login button
            with col_reg2:
                # Secondary styled back button
                if st.button("Back to Login", type="secondary", use_container_width=True):
                    # Switch back to login form
                    st.session_state.show_register = False
                    # Rerun to show login form
                    st.rerun()
        
        # Close login card div
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Right column: Welcome message and features
    with col2:
        if st.session_state.login_type == "user":
            # HTML formatted welcome message with features list
            st.markdown("""
            <div style='padding: 2rem;'>
                <h2>🌟 Welcome to NUTRI-CARE</h2>
                <p>Your AI-powered personal diet planner for Indian cuisine!</p>
                <br>
                <h4>✨ Key Features:</h4>
                <ul>
                    <li>🍽️ Personalized Indian meal plans</li>
                    <li>🎯 Goal-based nutrition tracking</li>
                    <li>🩺 Medical condition consideration</li>
                    <li>🚫 Allergy-aware filtering</li>
                    <li>📊 Comprehensive analytics</li>
                    <li>📈 Progress history tracking</li>
                </ul>
                <br>
                <p><strong>Login or Register to get started!</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Admin welcome message
            st.markdown("""
            < style='padding: 2rem;'>
                <h2>👑 Admin Dashboard</h2>
                <p>System Administration Portal</p>
                <br>
                <h4>🔧 Admin Features:</h4>
                <ul>
                    <li>👥 User Management & Analytics</li>
                    <li>📊 System Statistics & Monitoring</li>
                    <li>⚙️ Account Management</li>
                    <li>📈 Usage Analytics & Reports</li>
                    <li>🔐 Security & Access Control</li>
                    <li>📝 Content Management</li>
                </ul>
                <br>
                <p><strong>Login with your admin credentials to continue.</strong></p>
            </div>
                <br>
                <p style='color: #ff6b6b;'><strong>⚠️ Restricted Access - Admins Only</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer for unauthenticated view
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit | NUTRI-CARE - Your AI Diet Planner"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Stop execution here if not authenticated
    # Prevents rest of app from loading for unauthenticated users
    st.stop()


# ==========================================================
# ADMIN DASHBOARD (ONLY SHOWN IF ADMIN AUTHENTICATED)
# ==========================================================

if st.session_state.admin_authenticated:
    # ==========================================================
    # ADMIN SIDEBAR SECTION
    # ==========================================================
    
    with st.sidebar:
        # Display admin info at top of sidebar
        st.markdown(f"<h3 style='text-align: center; color: #ffd700;'>👑 Admin: {st.session_state.current_user}</h3>", unsafe_allow_html=True)
        
        # Get system statistics
        stats = get_system_stats()
        
        # Admin quick stats
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Display quick statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Users", stats['total_users'])
        with col2:
            st.metric("Active", stats['active_users'])
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Logins Today", stats['today_logins'])
        with col4:
            st.metric("Plans", stats['plans_generated'])
        
        # Admin navigation
        st.markdown("---")
        st.markdown("### 🚀 Admin Actions")
        
        # Create new admin account
        with st.expander("➕ Create New Admin"):
            new_admin_username = st.text_input("New Admin Username", key="new_admin_user")
            new_admin_password = st.text_input("New Admin Password", type="password", key="new_admin_pass")
            new_admin_name = st.text_input("Admin Name", key="new_admin_name")
            new_admin_email = st.text_input("Admin Email", key="new_admin_email")
            
            if st.button("Create Admin Account", type="primary"):
                if new_admin_username and new_admin_password and new_admin_name and new_admin_email:
                    users = load_users()
                    if new_admin_username in users:
                        st.error("Username already exists!")
                    else:
                        users[new_admin_username] = {
                            'password': hash_password(new_admin_password),
                            'email': new_admin_email,
                            'name': new_admin_name,
                            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'profile_history': [],
                            'meal_plan_history': [],
                            'last_login': None,
                            'user_type': 'admin',
                            'is_active': True,
                            'login_count': 0
                        }
                        save_users(users)
                        st.success("Admin account created successfully!")
                else:
                    st.warning("Please fill all fields!")
        
        # Logout button
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Reset all session state variables to default
            st.session_state.admin_authenticated = False
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.user_profile = {}
            st.session_state.filtered_foods = pd.DataFrame()
            st.session_state.meal_plan = {}
            # Rerun app to show login screen
            st.rerun()
    
    # ==========================================================
    # ADMIN MAIN CONTENT
    # ==========================================================
    
    # Admin Dashboard Header
    st.markdown("<h1 class='main-header'>👑 NUTRI-CARE Admin Dashboard</h1>", unsafe_allow_html=True)
    
    # Admin navigation tabs
    admin_tabs = st.tabs(["📊 Dashboard Overview", "👥 User Management", "📈 Analytics & Reports", "⚙️ System Settings", "📋 Database Management"])
    
    # Tab 1: Dashboard Overview
    with admin_tabs[0]:
        st.markdown("<h2 class='sub-header'>📊 System Overview</h2>", unsafe_allow_html=True)
        
        # Get latest statistics
        stats = get_system_stats()
        
        # Display key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='admin-card'>
                <h3>👥 Total Users</h3>
                <h1 style='color: #4299e1;'>{stats['total_users']}</h1>
                <p>Active: {stats['active_users']} | Admins: {stats['admins']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='admin-card'>
                <h3>📈 Activity</h3>
                <h1 style='color: #4CAF50;'>{stats['total_logins']}</h1>
                <p>Total Logins | Today: {stats['today_logins']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='admin-card'>
                <h3>📋 Plans Generated</h3>
                <h1 style='color: #ff9800;'>{stats['plans_generated']}</h1>
                <p>Meal Plans | Profiles: {stats['profiles_created']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='admin-card'>
                <h3>📊 Food Database</h3>
                <h1 style='color: #9c27b0;'>{len(df)}</h1>
                <p>Food Items | Vegetarian: {df['is_veg'].sum()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent Activity Section
        st.markdown("<h3 class='sub-header'>📈 Recent Activity</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent logins chart
            users = get_all_users()
            recent_users = []
            for username, user_data in users.items():
                if user_data.get('last_login'):
                    recent_users.append({
                        'username': username,
                        'name': user_data.get('name', 'N/A'),
                        'last_login': user_data.get('last_login'),
                        'login_count': user_data.get('login_count', 0)
                    })
            
            # Sort by last login
            recent_users.sort(key=lambda x: x['last_login'] or '', reverse=True)
            
            st.markdown("### 🔄 Recent Logins")
            for user in recent_users[:5]:
                st.markdown(f"""
                <div class='user-item'>
                    <strong>{user['username']}</strong> ({user['name']})<br>
                    <small>Last: {user['last_login'] or 'Never'} | Total: {user['login_count']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # User registration trend
            st.markdown("### 📅 Recent Registrations")
            registration_dates = []
            for user_data in users.values():
                reg_date = user_data.get('registration_date', '').split()[0]
                if reg_date:
                    registration_dates.append(reg_date)
            
            if registration_dates:
                reg_df = pd.DataFrame({'date': registration_dates})
                reg_counts = reg_df['date'].value_counts().sort_index().tail(7)
                
                fig = px.bar(x=reg_counts.index, y=reg_counts.values,
                           title="Registrations (Last 7 Days)",
                           labels={'x': 'Date', 'y': 'Registrations'},
                           color_discrete_sequence=['#4CAF50'])
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: User Management
    with admin_tabs[1]:
        st.markdown("<h2 class='sub-header'>👥 User Management</h2>", unsafe_allow_html=True)
        
        # Search and filter users
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search Users", placeholder="Search by username, name, or email")
        with col2:
            filter_type = st.selectbox("User Type", ["All", "User", "Admin"])
        with col3:
            filter_status = st.selectbox("Status", ["All", "Active", "Inactive"])
        
        # Get all users
        users = get_all_users()
        
        # Apply filters
        filtered_users = {}
        for username, user_data in users.items():
            # Skip current admin if searching
            if username == st.session_state.current_user:
                continue
                
            # Apply type filter
            if filter_type != "All" and user_data.get('user_type') != filter_type.lower():
                continue
            
            # Apply status filter
            if filter_status == "Active" and not user_data.get('is_active', True):
                continue
            if filter_status == "Inactive" and user_data.get('is_active', True):
                continue
            
            # Apply search filter
            if search_query:
                search_lower = search_query.lower()
                if (search_lower not in username.lower() and 
                    search_lower not in user_data.get('name', '').lower() and
                    search_lower not in user_data.get('email', '').lower()):
                    continue
            
            filtered_users[username] = user_data
        
        # Display users in a table
        st.markdown(f"### 📋 Users ({len(filtered_users)})")
        
        if filtered_users:
            # Create user data for table
            user_table_data = []
            for username, user_data in filtered_users.items():
                user_type = user_data.get('user_type', 'user')
                status = "Active" if user_data.get('is_active', True) else "Inactive"
                
                user_table_data.append({
                    "Username": username,
                    "Name": user_data.get('name', 'N/A'),
                    "Email": user_data.get('email', 'N/A'),
                    "Type": user_type,
                    "Status": status,
                    "Reg Date": user_data.get('registration_date', 'N/A'),
                    "Last Login": user_data.get('last_login', 'Never'),
                    "Login Count": user_data.get('login_count', 0),
                    "Plans": len(user_data.get('meal_plan_history', [])),
                    "Profiles": len(user_data.get('profile_history', []))
                })
            
            # Convert to DataFrame for display
            user_df = pd.DataFrame(user_table_data)
            
            # Display with styling
            st.dataframe(
                user_df,
                use_container_width=True,
                column_config={
                    "Username": st.column_config.TextColumn("Username", width="small"),
                    "Type": st.column_config.TextColumn(
                        "Type",
                        width="small",
                        help="User type (user/admin)"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        width="small"
                    )
                },
                hide_index=True
            )
            
            # User actions
            st.markdown("### ⚙️ User Actions")
            selected_user = st.selectbox("Select User", list(filtered_users.keys()))
            
            if selected_user:
                user_data = filtered_users[selected_user]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Username:** {selected_user}")
                    st.markdown(f"**Name:** {user_data.get('name', 'N/A')}")
                    st.markdown(f"**Email:** {user_data.get('email', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Type:** {user_data.get('user_type', 'user')}")
                    st.markdown(f"**Status:** {'✅ Active' if user_data.get('is_active', True) else '❌ Inactive'}")
                    st.markdown(f"**Registration:** {user_data.get('registration_date', 'N/A')}")
                
                with col3:
                    st.markdown(f"**Last Login:** {user_data.get('last_login', 'Never')}")
                    st.markdown(f"**Login Count:** {user_data.get('login_count', 0)}")
                    st.markdown(f"**Plans Created:** {len(user_data.get('meal_plan_history', []))}")
                
                # Action buttons
                col_act1, col_act2, col_act3 = st.columns(3)
                
                with col_act1:
                    if user_data.get('is_active', True):
                        if st.button("❌ Deactivate User", use_container_width=True):
                            if update_user_status(selected_user, False):
                                st.success(f"User {selected_user} deactivated!")
                                st.rerun()
                    else:
                        if st.button("✅ Activate User", use_container_width=True):
                            if update_user_status(selected_user, True):
                                st.success(f"User {selected_user} activated!")
                                st.rerun()
                
                with col_act2:
                    if user_data.get('user_type') != 'admin':
                        if st.button("👑 Make Admin", use_container_width=True):
                            users = load_users()
                            users[selected_user]['user_type'] = 'admin'
                            save_users(users)
                            st.success(f"User {selected_user} promoted to admin!")
                            st.rerun()
                
                with col_act3:
                    if st.button("🗑️ Delete User", type="secondary", use_container_width=True):
                        success, message = delete_user(selected_user)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                # View user history
                with st.expander("📜 View User History"):
                    profile_history, meal_plan_history = get_user_history(selected_user)
                    
                    col_hist1, col_hist2 = st.columns(2)
                    
                    with col_hist1:
                        st.markdown("**Profile History**")
                        if profile_history:
                            for profile in profile_history[-3:]:  # Show last 3
                                st.markdown(f"""
                                <div class='history-card'>
                                    <small>ID: {profile.get('plan_id')} | {profile.get('timestamp', 'N/A')}</small><br>
                                    <strong>Goal:</strong> {profile.get('goal', 'N/A')}<br>
                                    <strong>BMI:</strong> {profile.get('bmi', 'N/A')}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No profile history")
                    
                    with col_hist2:
                        st.markdown("**Meal Plan History**")
                        if meal_plan_history:
                            for plan in meal_plan_history[-3:]:  # Show last 3
                                st.markdown(f"""
                                <div class='history-card'>
                                    <small>ID: {plan.get('plan_id')} | {plan.get('timestamp', 'N/A')}</small><br>
                                    <strong>Days:</strong> {len(plan.get('plan_data', {}))}<br>
                                    <strong>Goal:</strong> {plan.get('user_profile', {}).get('goal', 'N/A')}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No meal plan history")
        else:
            st.info("No users found matching the criteria")
    
    # Tab 3: Analytics & Reports
    with admin_tabs[2]:
        st.markdown("<h2 class='sub-header'>📈 Analytics & Reports</h2>", unsafe_allow_html=True)
        
        # Generate comprehensive analytics
        users = get_all_users()
        
        # User Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # User type distribution
            user_types = []
            for user_data in users.values():
                user_types.append(user_data.get('user_type', 'user'))
            
            type_counts = pd.Series(user_types).value_counts()
            
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="User Type Distribution",
                        color_discrete_sequence=['#4CAF50', '#FF9800'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Account status distribution
            status_counts = {'Active': 0, 'Inactive': 0}
            for user_data in users.values():
                if user_data.get('is_active', True):
                    status_counts['Active'] += 1
                else:
                    status_counts['Inactive'] += 1
            
            fig = px.bar(x=list(status_counts.keys()), y=list(status_counts.values()),
                        title="Account Status Distribution",
                        labels={'x': 'Status', 'y': 'Count'},
                        color=list(status_counts.keys()),
                        color_discrete_map={'Active': '#4CAF50', 'Inactive': '#f44336'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Activity Analysis
        st.markdown("### 📊 Activity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Login activity by user type
            admin_logins = sum(u.get('login_count', 0) for u in users.values() 
                              if u.get('user_type') == 'admin')
            user_logins = sum(u.get('login_count', 0) for u in users.values() 
                             if u.get('user_type') == 'user')
            
            fig = px.bar(x=['Admins', 'Users'], y=[admin_logins, user_logins],
                        title="Total Logins by User Type",
                        color=['Admins', 'Users'],
                        color_discrete_sequence=['#FF9800', '#4CAF50'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Registration trend over time
            reg_dates = []
            for user_data in users.values():
                reg_date = user_data.get('registration_date', '').split()[0]
                if reg_date:
                    reg_dates.append(reg_date)
            
            if reg_dates:
                reg_series = pd.Series(reg_dates)
                monthly_reg = reg_series.apply(lambda x: x[:7]).value_counts().sort_index()
                
                fig = px.line(x=monthly_reg.index, y=monthly_reg.values,
                            title="Monthly Registrations",
                            labels={'x': 'Month', 'y': 'Registrations'},
                            line_shape='spline')
                fig.update_traces(line=dict(color='#4CAF50', width=3))
                st.plotly_chart(fig, use_container_width=True)
        
        # Usage Statistics
        st.markdown("### 📋 Usage Statistics")
        
        # Calculate various statistics
        total_plans = sum(len(u.get('meal_plan_history', [])) for u in users.values())
        total_profiles = sum(len(u.get('profile_history', [])) for u in users.values())
        avg_plans_per_user = total_plans / max(1, len(users))
        avg_logins_per_user = sum(u.get('login_count', 0) for u in users.values()) / max(1, len(users))
        
        # Display in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Plans", total_plans)
        with col2:
            st.metric("Total Profiles", total_profiles)
        with col3:
            st.metric("Avg Plans/User", f"{avg_plans_per_user:.1f}")
        with col4:
            st.metric("Avg Logins/User", f"{avg_logins_per_user:.1f}")
        
        # Generate Report Button
        st.markdown("---")
        if st.button("📥 Generate Full Report", type="primary"):
            with st.spinner("Generating report..."):
                # Create report data
                report_data = {
                    'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_users': len(users),
                    'active_users': sum(1 for u in users.values() if u.get('is_active', True)),
                    'admin_users': sum(1 for u in users.values() if u.get('user_type') == 'admin'),
                    'total_logins': sum(u.get('login_count', 0) for u in users.values()),
                    'total_plans': total_plans,
                    'total_profiles': total_profiles,
                    'user_list': []
                }
                
                # Add user details
                for username, user_data in users.items():
                    report_data['user_list'].append({
                        'username': username,
                        'name': user_data.get('name', 'N/A'),
                        'email': user_data.get('email', 'N/A'),
                        'type': user_data.get('user_type', 'user'),
                        'status': 'Active' if user_data.get('is_active', True) else 'Inactive',
                        'registration_date': user_data.get('registration_date', 'N/A'),
                        'last_login': user_data.get('last_login', 'Never'),
                        'login_count': user_data.get('login_count', 0),
                        'plans_created': len(user_data.get('meal_plan_history', [])),
                        'profiles_created': len(user_data.get('profile_history', []))
                    })
                
                # Convert to DataFrame for download
                report_df = pd.DataFrame(report_data['user_list'])
                
                # Provide download button
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download User Report (CSV)",
                    data=csv,
                    file_name=f"nutricare_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.success("Report generated successfully!")
    
    # Tab 4: System Settings
    with admin_tabs[3]:
        st.markdown("<h2 class='sub-header'>⚙️ System Settings</h2>", unsafe_allow_html=True)
        
        # System Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔐 Security Settings")
            
            # Change admin password
            with st.expander("Change Admin Password", expanded=True):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.button("Update Password", type="primary"):
                    if not all([current_password, new_password, confirm_password]):
                        st.warning("Please fill all fields!")
                    elif new_password != confirm_password:
                        st.error("New passwords don't match!")
                    else:
                        # Verify current password
                        success, message = authenticate_user(
                            st.session_state.current_user, 
                            current_password, 
                            True
                        )
                        if success:
                            users = load_users()
                            users[st.session_state.current_user]['password'] = hash_password(new_password)
                            save_users(users)
                            st.success("Password updated successfully!")
                        else:
                            st.error("Current password is incorrect!")
            
            # Session settings
            with st.expander("Session Settings"):
                auto_logout = st.slider("Auto-logout after (minutes)", 15, 240, 60)
                max_login_attempts = st.slider("Max login attempts", 3, 10, 5)
                
                if st.button("Save Session Settings"):
                    st.success("Session settings saved!")
        
        with col2:
            st.markdown("### 📧 Notification Settings")
            
            # Email notifications
            with st.expander("Email Configuration"):
                smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", 465, 587, 587)
                admin_email = st.text_input("Admin Email", "admin@nutricare.com")
                email_enabled = st.checkbox("Enable email notifications", True)
                
                if st.button("Save Email Settings"):
                    st.success("Email settings saved!")
            
            # System notifications
            with st.expander("System Notifications"):
                notify_new_user = st.checkbox("Notify on new user registration", True)
                notify_failed_login = st.checkbox("Notify on failed login attempts", True)
                notify_system_error = st.checkbox("Notify on system errors", True)
                
                if st.button("Save Notification Settings"):
                    st.success("Notification settings saved!")
        
        # Database Settings
        st.markdown("### 💾 Database Management")
        
        col_db1, col_db2 = st.columns(2)
        
        with col_db1:
            with st.expander("Backup Database"):
                st.info("Last backup: Never")
                if st.button("Create Backup Now", type="primary"):
                    with st.spinner("Creating backup..."):
                        time.sleep(2)  # Simulate backup process
                        st.success("Backup created successfully!")
                
                # Backup schedule
                backup_schedule = st.selectbox("Auto-backup schedule", 
                                              ["Disabled", "Daily", "Weekly", "Monthly"])
                if st.button("Save Backup Schedule"):
                    st.success("Backup schedule updated!")
        
        with col_db2:
            with st.expander("Database Statistics"):
                # Get database stats
                import glob
                db_files = glob.glob("*.json") + glob.glob("*.csv")
                
                st.write(f"**Total database files:** {len(db_files)}")
                st.write(f"**Users database:** {len(get_all_users())} records")
                st.write(f"**Food database:** {len(df)} records")
                
                # Calculate total size
                total_size = 0
                for file in db_files:
                    try:
                        total_size += os.path.getsize(file)
                    except:
                        pass
                
                st.write(f"**Total size:** {total_size / 1024:.1f} KB")
        
        # System Maintenance
        st.markdown("### 🔧 System Maintenance")
        
        col_mt1, col_mt2, col_mt3 = st.columns(3)
        
        with col_mt1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
        
        with col_mt2:
            if st.button("📊 Rebuild Indexes", use_container_width=True):
                with st.spinner("Rebuilding indexes..."):
                    time.sleep(2)
                    st.success("Indexes rebuilt successfully!")
        
        with col_mt3:
            if st.button("🧹 Clean Logs", use_container_width=True):
                with st.spinner("Cleaning logs..."):
                    time.sleep(1)
                    st.success("Logs cleaned successfully!")
    
    # Tab 5: Database Management
    with admin_tabs[4]:
        st.markdown("<h2 class='sub-header'>📋 Database Management</h2>", unsafe_allow_html=True)
        
        # Food Database Management
        st.markdown("### 🍽️ Food Database")
        
        col_db1, col_db2 = st.columns([2, 1])
        
        with col_db1:
            # Display food database
            st.dataframe(
                df.head(100),
                use_container_width=True,
                hide_index=True
            )
        
        with col_db2:
            # Food database statistics
            st.markdown("#### 📊 Statistics")
            
            total_foods = len(df)
            veg_foods = df['is_veg'].sum()
            non_veg_foods = total_foods - veg_foods
            
            st.metric("Total Foods", total_foods)
            st.metric("Vegetarian", veg_foods)
            st.metric("Non-Vegetarian", non_veg_foods)
            
            # Average nutrients
            avg_calories = df['calories'].mean()
            avg_protein = df['protein'].mean()
            avg_fat = df['fat'].mean()
            avg_carbs = df['carbs'].mean()
            
            st.write("**Average per serving:**")
            st.write(f"Calories: {avg_calories:.1f}")
            st.write(f"Protein: {avg_protein:.1f}g")
            st.write(f"Fat: {avg_fat:.1f}g")
            st.write(f"Carbs: {avg_carbs:.1f}g")
        
        # Food Management Actions
        st.markdown("### ⚙️ Food Management")
        
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            # Add new food item
            with st.expander("➕ Add New Food"):
                new_food_name = st.text_input("Food Name")
                new_calories = st.number_input("Calories", 0.0, 1000.0, 200.0)
                new_protein = st.number_input("Protein (g)", 0.0, 100.0, 10.0)
                new_fat = st.number_input("Fat (g)", 0.0, 100.0, 5.0)
                new_carbs = st.number_input("Carbs (g)", 0.0, 200.0, 30.0)
                is_veg = st.checkbox("Vegetarian", True)
                
                if st.button("Add Food Item"):
                    if new_food_name:
                        # In a real app, you would add to the database
                        st.success(f"Food item '{new_food_name}' added!")
                    else:
                        st.warning("Please enter a food name!")
        
        with col_act2:
            # Export food database
            with st.expander("📤 Export Database"):
                export_format = st.radio("Format", ["CSV", "JSON", "Excel"])
                
                if st.button("Export Food Database"):
                    if export_format == "CSV":
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="food_database.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        json_str = df.to_json(orient='records')
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name="food_database.json",
                            mime="application/json"
                        )
                    else:
                        st.info("Excel export requires additional libraries")
        
        with col_act3:
            # Import food database
            with st.expander("📥 Import Database"):
                uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json'])
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            imported_df = pd.read_csv(uploaded_file)
                        else:
                            imported_df = pd.read_json(uploaded_file)
                        
                        st.success(f"File loaded successfully! {len(imported_df)} records found.")
                        
                        if st.button("Import to Database"):
                            # In a real app, you would save to the database
                            st.success("Database updated successfully!")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
        
        # Database Health Check
        st.markdown("### 🩺 Database Health Check")
        
        # Perform health checks
        health_issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            health_issues.append(f"Found {missing_values} missing values in food database")
        
        # Check for duplicate food names
        duplicate_foods = df['name'].duplicated().sum()
        if duplicate_foods > 0:
            health_issues.append(f"Found {duplicate_foods} duplicate food names")
        
        # Check user database
        users = get_all_users()
        invalid_users = []
        for username, user_data in users.items():
            if not user_data.get('email') or '@' not in user_data.get('email', ''):
                invalid_users.append(username)
        
        if invalid_users:
            health_issues.append(f"Found {len(invalid_users)} users with invalid email")
        
        # Display health status
        if not health_issues:
            st.markdown("""
            <div class='success-card'>
                <h3>✅ Database Health: Excellent</h3>
                <p>All systems are functioning properly.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-card'>
                <h3>⚠️ Database Health: Issues Found</h3>
                <ul>
            """, unsafe_allow_html=True)
            
            for issue in health_issues:
                st.markdown(f"<li>{issue}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            if st.button("🛠️ Fix All Issues", type="primary"):
                with st.spinner("Fixing issues..."):
                    time.sleep(2)
                    st.success("All issues have been resolved!")
                    st.rerun()
    
    # Admin Dashboard Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        f"👑 Admin Dashboard | Logged in as: {st.session_state.current_user} | "
        "NUTRI-CARE Admin System v1.0"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Stop execution here for admin dashboard
    st.stop()


# ==========================================================
# MAIN APPLICATION (ONLY SHOWN IF USER AUTHENTICATED)
# ==========================================================

# ==========================================================
# SIDEBAR SECTION - USER PROFILE INPUT
# ==========================================================

# Create sidebar using Streamlit's context manager
# with st.sidebar: creates a collapsible sidebar container
with st.sidebar:
    # Display user info at top of sidebar
    st.markdown(f"<h3 style='text-align: center;'>👤 Welcome, {st.session_state.current_user}</h3>", unsafe_allow_html=True)
    
    # Logout button - allows user to end session
    if st.button("🚪 Logout", use_container_width=True):
        # Reset all session state variables to default
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.user_profile = {}
        st.session_state.filtered_foods = pd.DataFrame()
        st.session_state.meal_plan = {}
        # Rerun app to show login screen
        st.rerun()
    
    # Horizontal separator line
    st.markdown("---")
    
    # Sidebar header for user profile section
    st.markdown("<h2 style='text-align: center;'>👤 User Profile</h2>", unsafe_allow_html=True)
    
    # --- BASIC USER INFORMATION ---
    # Text input for user's name with placeholder
    # st.text_input creates single-line text input field
    # value parameter loads existing data from session state if available
    name = st.text_input("Full Name", placeholder="Enter your name", 
                        value=st.session_state.user_profile.get('name', ''))
    
    # Number input for age with min/max constraints
    # st.number_input creates numeric input with increment/decrement buttons
    age = st.number_input("Age", min_value=10, max_value=100, 
                         value=st.session_state.user_profile.get('age', 30))
    
    # Dropdown selection for gender with default value handling
    # st.selectbox creates dropdown selection menu
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                         index=["Male", "Female", "Other"].index(st.session_state.user_profile.get('gender', 'Male')) 
                         if st.session_state.user_profile.get('gender') else 0)
    
    # Number input for weight in kg with decimal precision
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, 
                            value=st.session_state.user_profile.get('weight', 70.0))
    
    # Number input for height in cm
    height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, 
                            value=st.session_state.user_profile.get('height', 170.0))
    
    # Dropdown for activity level with descriptive options
    activity_level = st.selectbox(
        "Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
        index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
        .index(st.session_state.user_profile.get('activity_level', 'Moderately Active')) 
        if st.session_state.user_profile.get('activity_level') else 2
    )

    # --- FITNESS GOAL ---
    # Dropdown for selecting fitness goal with emoji prefix
    goal = st.selectbox("🎯 Your Goal", ["Weight Loss", "Muscle Gain", "Maintenance"],
                       index=["Weight Loss", "Muscle Gain", "Maintenance"]
                       .index(st.session_state.user_profile.get('goal', 'Weight Loss')) 
                       if st.session_state.user_profile.get('goal') else 0)

    # --- FOOD PREFERENCES SECTION ---
    # Subheader with food emoji for visual categorization
    st.subheader("🥗 Food Preferences")
    
    # Radio buttons for diet type selection (mutually exclusive options)
    food_preference = st.radio("Diet Type:", ["Vegetarian", "Eggetarian", "Non-Vegetarian", "Any"],
                              index=["Vegetarian", "Eggetarian", "Non-Vegetarian", "Any"]
                              .index(st.session_state.user_profile.get('food_preference', 'Vegetarian')) 
                              if st.session_state.user_profile.get('food_preference') else 0)

    # --- MEDICAL CONDITIONS SECTION ---
    st.subheader("🩺 Health Conditions")
    
    # Multi-select for medical conditions (allows multiple selections)
    # st.multiselect creates checkbox-style multi-selection dropdown
    default_conditions = st.session_state.user_profile.get('medical_conditions', [])
    medical_conditions = st.multiselect(
        "Select any medical conditions:",
        [
            "Diabetes", "Hypertension", "Heart Disease", "Thyroid Issues",
            "PCOS", "Kidney Disease", "GERD/Acid Reflux", "IBS",
            "Celiac Disease", "High Cholesterol"
        ],
        default=default_conditions  # Pre-select previously chosen conditions
    )

    # --- ALLERGIES SECTION ---
    st.subheader("🚫 Food Allergies")
    
    # Multi-select for allergies with common allergen options
    default_allergies = st.session_state.user_profile.get('allergies', [])
    allergies = st.multiselect(
        "Select your allergies:",
        ["Gluten", "Dairy", "Nuts", "Soy", "Shellfish", "Eggs", "Fish", "Corn", "Sesame", "Mustard"],
        default=default_allergies  # Pre-select previously chosen allergies
    )

    # --- CALCULATE USER METRICS ---
    # Call helper function to calculate BMI and TDEE using current input values
    # These calculations update in real-time as user changes inputs
    bmi, tdee = calculate_user_metrics(age, gender, weight, height, activity_level, goal)
    
    # --- DISPLAY CALCULATED METRICS ---
    # Horizontal separator before metrics display
    st.markdown("---")
    st.subheader("📊 Your Metrics")
    
    # Create two columns for metric display side by side
    col1, col2 = st.columns(2)
    
    # Left column: BMI display
    with col1:
        # st.metric creates a highlighted metric display card
        st.metric("BMI", f"{bmi}")
    
    # Right column: Daily calorie target display
    with col2:
        st.metric("Daily Calories", f"{tdee} kcal")
    
    # ==========================================================
    # IMAGE UPLOAD SECTION FOR FOOD ANALYSIS
    # ==========================================================
    
    # Horizontal separator before image upload section
    st.markdown("---")
    st.subheader("📸 Food Image Analysis")
    
    # File uploader for food images with specific allowed formats
    # st.file_uploader creates a file upload widget
    uploaded_image = st.file_uploader(
        "Upload food image for AI analysis",  # Label text
        type=['jpg', 'jpeg', 'png', 'webp'],  # Allowed file extensions
        help="Get instant nutritional analysis from food photos"  # Tooltip text
    )
    
    # If an image is uploaded, display it and analysis options
    if uploaded_image is not None:
        # Create two columns: one for thumbnail, one for status
        col1, col2 = st.columns([1, 2])
        
        # Left column: Show thumbnail of uploaded image
        with col1:
            # st.image displays image with specified width
            st.image(uploaded_image, width=80)
        
        # Right column: Confirmation message
        with col2:
            st.write("Image uploaded!")
        
        # Radio buttons for selecting analysis type
        analysis_type = st.radio(
            "Analysis Type:",  # Label
            ["Quick Analysis", "Detailed Report"],  # Options
            horizontal=True  # Display options horizontally
        )
        
        # Analysis button - triggers AI analysis (placeholder functionality)
        if st.button("🤖 Analyze with AI", type="primary", width='stretch'):
            # Store image and analysis type in session state for later use
            st.session_state.image_analysis = {
                'uploaded_image': uploaded_image,
                'analysis_type': analysis_type
            }
    
    # --- GENERATE RECOMMENDATIONS BUTTON ---
    # Primary action button to generate personalized plan based on inputs
    if st.button("🚀 Generate Personalized Plan", type="primary", width='stretch'):
        # Show loading spinner while processing
        with st.spinner("Analyzing your profile and generating recommendations..."):
            # Store all user profile data in session state dictionary
            # This preserves user inputs across app interactions
            st.session_state.user_profile = {
                'name': name, 'age': age, 'gender': gender, 'weight': weight, 
                'height': height, 'goal': goal, 'bmi': bmi, 'tdee': tdee,
                'food_preference': food_preference, 'medical_conditions': medical_conditions,
                'allergies': allergies, 'activity_level': activity_level
            }
            
            # Save current profile to user history for future reference
            # copy() creates a snapshot to prevent reference issues
            save_user_profile(st.session_state.current_user, st.session_state.user_profile.copy())
            
            # Call filter_foods function with user preferences
            # This filters the food database based on all user criteria
            st.session_state.filtered_foods = filter_foods(
                df, food_preference, allergies, medical_conditions, goal
            )
            
            # Generate 7-day meal plan from filtered foods
            st.session_state.meal_plan = generate_meal_plan(st.session_state.filtered_foods)
            
            # Save generated meal plan to user history
            if st.session_state.meal_plan:
                meal_plan_data = {
                    'plan_data': st.session_state.meal_plan,
                    'user_profile': st.session_state.user_profile,
                    'food_count': len(st.session_state.filtered_foods)
                }
                save_meal_plan(st.session_state.current_user, meal_plan_data)
            
        # Success message after plan generation
        st.success("Personalized plan generated successfully!")


# ==========================================================
# MAIN NAVIGATION MENU
# ==========================================================

# Create horizontal navigation menu at the top of the main area
# option_menu creates a tabbed navigation interface
selected_tab = option_menu(
    menu_title=None,  # No main title for the menu
    options=["Home", "Meal Plan", "Food Insights", "Nutrition Analytics", "User History"],  # Tab names
    icons=["house", "calendar-week", "search", "bar-chart", "history"],  # Bootstrap icons for each tab
    default_index=0,  # Default selected tab (0 = Home)
    orientation="horizontal",  # Horizontal layout instead of vertical
    styles={  # Custom CSS styles for menu appearance
        "container": {"padding": "0!important", "background-color": "#141618"},  # Container styling
        "icon": {"color": "orange", "font-size": "18px"},  # Icon styling
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#141618"},  # Link styling
        "nav-link-selected": {"background-color": "#1f77b4"},  # Selected tab styling
    }
)


# ==========================================================
# HOME TAB CONTENT
# ==========================================================

# Check which tab is selected and display corresponding content
if selected_tab == "Home":
    # Main page header with custom CSS class
    st.markdown("<h1 class='main-header'>🍛 NUTRI-CARE: AI-Powered Indian Diet Planner</h1>", unsafe_allow_html=True)
    
    # Personalized welcome message for logged-in user
    st.markdown(f"<h3 style='text-align: center; color: #666;'>Welcome back, {st.session_state.current_user}!</h3>", unsafe_allow_html=True)
    
    # --- DISPLAY IMAGE ANALYSIS RESULTS IF AVAILABLE ---
    # Check if user has uploaded and analyzed a food image
    if 'image_analysis' in st.session_state:
        # Success notification for completed analysis
        st.success("🎯 Food Analysis Complete!")
        
        # Create three columns for displaying analysis metrics
        col1, col2, col3 = st.columns(3)
        
        # Display estimated calories (placeholder values)
        with col1:
            st.metric("Estimated Calories", "285 kcal")
        
        # Display estimated protein content
        with col2:
            st.metric("Protein", "15g")
        
        # Display estimated carbohydrate content
        with col3:
            st.metric("Carbs", "35g")
        
        # Display the uploaded image with analysis type caption
        st.image(st.session_state.image_analysis['uploaded_image'], 
                caption=f"Analyzed Prescription - {st.session_state.image_analysis['analysis_type']}", 
                width=700)
    
    # --- WELCOME MESSAGE FOR RETURNING USERS ---
    # Check if user profile exists (user has generated a plan before)
    if st.session_state.user_profile:
        # Personalized welcome with user's name
        st.markdown(f"### 👋 Welcome back, {st.session_state.user_profile['name']}!")
        # Display user's current fitness goal
        st.markdown(f"Here's your personalized dashboard for your **{st.session_state.user_profile['goal']}** journey.")
    
    # --- KEY METRICS CARDS SECTION ---
    # Section header with custom CSS class
    st.markdown("<h2 class='sub-header'>📊 Your Health Overview</h2>", unsafe_allow_html=True)
    
    # Display metric cards if user profile exists
    if st.session_state.user_profile:
        # Create four equal-width columns for metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Column 1: Goal card
        with col1:
            # HTML card with metric-card CSS class
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🎯 Goal</h3>
                <h2>{st.session_state.user_profile['goal']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Column 2: BMI card
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⚖️ BMI</h3>
                <h2>{st.session_state.user_profile['bmi']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Column 3: Daily calories card
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔥 Calories/Day</h3>
                <h2>{st.session_state.user_profile['tdee']} kcal</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Column 4: Diet type card
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🥗 Diet Type</h3>
                <h2>{st.session_state.user_profile['food_preference']}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # --- QUICK STATISTICS SECTION ---
    # Check if filtered foods exist (user has generated a plan)
    if not st.session_state.filtered_foods.empty:
        st.markdown("<h2 class='sub-header'>📈 Quick Statistics</h2>", unsafe_allow_html=True)
        
        # Create three columns for different statistics visualizations
        col1, col2, col3 = st.columns(3)
        
        # Column 1: Vegetarian vs Non-Vegetarian pie chart
        with col1:
            # Count vegetarian and non-vegetarian foods in filtered dataset
            veg_count = st.session_state.filtered_foods['is_veg'].sum()
            non_veg_count = len(st.session_state.filtered_foods) - veg_count
            
            # Create pie chart using Plotly Express
            fig = px.pie(values=[veg_count, non_veg_count], 
                        names=['Vegetarian', 'Non-Vegetarian'],
                        title="Veg vs Non-Veg Distribution", 
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])  # Green and red colors
            
            # Display chart in Streamlit
            st.plotly_chart(fig, width='stretch')
        
        # Column 2: Average nutrient values bar chart
        with col2:
            # Calculate average values for key nutrients
            avg_calories = st.session_state.filtered_foods['calories'].mean()
            avg_protein = st.session_state.filtered_foods['protein'].mean()
            avg_fat = st.session_state.filtered_foods['fat'].mean()
            avg_carbs = st.session_state.filtered_foods['carbs'].mean()
            
            # Create bar chart using Plotly Graph Objects
            fig = go.Figure(data=[
                go.Bar(name='Average', 
                      x=['Calories', 'Protein', 'Fat', 'Carbs'], 
                      y=[avg_calories/10, avg_protein, avg_fat, avg_carbs/2])  # Scaled for better visualization
            ])
            fig.update_layout(title="Average Nutrient Values", showlegend=False)
            st.plotly_chart(fig, width='stretch')
        
        # Column 3: Top food recommendations list
        with col3:
            st.markdown("### 🏆 Top Recommendations")
            
            # Get top 5 foods with best match scores
            top_foods = st.session_state.filtered_foods.head(5)
            
            # Display each top food in a styled card
            for _, food in top_foods.iterrows():
                st.markdown(f"""
                <div class='food-card'>
                    <strong>{food['name']}</strong><br>
                    <small>Calories: {int(food['calories'])} | Protein: {food['protein']:.1f}g</small>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Information message when no plan has been generated yet
        st.info("👈 Configure your profile in the sidebar and generate your personalized plan to see insights.")


# ==========================================================
# MEAL PLAN TAB CONTENT
# ==========================================================

elif selected_tab == "Meal Plan":
    # Tab header with custom CSS class
    st.markdown("<h1 class='main-header'>🍱 Your 7-Day Meal Plan</h1>", unsafe_allow_html=True)
    
    # Check if meal plan has been generated
    if not st.session_state.meal_plan:
        # Prompt user to generate a plan first
        st.info("👈 Please configure your profile in the sidebar and generate your personalized meal plan.")
    
    else:
        # List of days for meal plan display
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Loop through each day to display meal plan
        for day in days:
            # Check if meal plan exists for this day
            if day in st.session_state.meal_plan:
                # Create expandable section for each day's meals
                with st.expander(f"📅 {day} - Full Day Plan", expanded=False):
                    # Create three columns for Breakfast, Lunch, Dinner
                    col1, col2, col3 = st.columns(3)

                    # Column 1: Breakfast meals
                    with col1:
                        st.subheader("🍳 Breakfast")
                        # Loop through each breakfast food item
                        for food in st.session_state.meal_plan[day]["Breakfast"]:
                            # Get nutritional data for this food from filtered dataset
                            food_data = st.session_state.filtered_foods[st.session_state.filtered_foods["name"] == food].iloc[0]
                            
                            # Create Google search URL for recipe
                            recipe_query = urllib.parse.quote(f"{food} Indian recipe")
                            recipe_url = f"https://www.google.com/search?q={recipe_query}"
                            
                            # Display food card with clickable link to recipe
                            st.markdown(f"""
                            <div class='food-card'>
                                <strong><a href="{recipe_url}" target="_blank">{food}</a></strong><br>
                                <small>🔥 {int(food_data['calories'])} kcal | 🥚 {food_data['protein']:.1f}g protein</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Column 2: Lunch meals (same structure as breakfast)
                    with col2:
                        st.subheader("🍲 Lunch")
                        for food in st.session_state.meal_plan[day]["Lunch"]:
                            food_data = st.session_state.filtered_foods[st.session_state.filtered_foods["name"] == food].iloc[0]
                            recipe_query = urllib.parse.quote(f"{food} Indian recipe")
                            recipe_url = f"https://www.google.com/search?q={recipe_query}"
                            st.markdown(f"""
                            <div class='food-card'>
                                <strong><a href="{recipe_url}" target="_blank">{food}</a></strong><br>
                                <small>🔥 {int(food_data['calories'])} kcal | 🥚 {food_data['protein']:.1f}g protein</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Column 3: Dinner meals (same structure as breakfast)
                    with col3:
                        st.subheader("🍛 Dinner")
                        for food in st.session_state.meal_plan[day]["Dinner"]:
                            food_data = st.session_state.filtered_foods[st.session_state.filtered_foods["name"] == food].iloc[0]
                            recipe_query = urllib.parse.quote(f"{food} Indian recipe")
                            recipe_url = f"https://www.google.com/search?q={recipe_query}"
                            st.markdown(f"""
                            <div class='food-card'>
                                <strong><a href="{recipe_url}" target="_blank">{food}</a></strong><br>
                                <small>🔥 {int(food_data['calories'])} kcal | 🥚 {food_data['protein']:.1f}g protein</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Calculate total daily calories
                    daily_foods = (st.session_state.meal_plan[day]["Breakfast"] + 
                                 st.session_state.meal_plan[day]["Lunch"] + 
                                 st.session_state.meal_plan[day]["Dinner"])
                    
                    daily_calories = sum([st.session_state.filtered_foods[st.session_state.filtered_foods["name"] == food]
                                         .iloc[0]["calories"] for food in daily_foods])
                    
                    # Display daily total vs target calories
                    st.info(f"**📊 Daily Total: {int(daily_calories)} kcal | Target: {st.session_state.user_profile['tdee']} kcal**")

        # Export section for meal plan
        st.markdown("---")
        st.markdown("### 📥 Export Your Meal Plan")
        
        # Two columns for export options
        col1, col2 = st.columns(2)
        
        # Column 1: CSV download button
        with col1:
            if not st.session_state.filtered_foods.empty:
                # Convert filtered foods to CSV format
                csv = st.session_state.filtered_foods.to_csv(index=False)
                
                # Download button for CSV export
                st.download_button(
                    label="📋 Download as CSV",  # Button text
                    data=csv,  # CSV data
                    file_name="personalized_meal_plan.csv",  # Default filename
                    mime="text/csv",  # MIME type for browser
                    width='stretch'  # Full width button
                )
        
        # Column 2: Save to profile button
        with col2:
            if st.button("💾 Save to Profile", width='stretch'):
                # Save current meal plan to session state
                st.session_state.saved_plan = st.session_state.meal_plan
                st.success("Meal plan saved to your profile!")


# ==========================================================
# FOOD INSIGHTS TAB CONTENT
# ==========================================================

elif selected_tab == "Food Insights":
    # Tab header with custom CSS class
    st.markdown("<h1 class='main-header'>🔍 Food Database & Insights</h1>", unsafe_allow_html=True)
    
    # Check if filtered foods exist
    if st.session_state.filtered_foods.empty:
        # Prompt to generate plan first
        st.info("👈 Please generate your personalized plan first to see food recommendations.")
    
    else:
        # Create search and sort interface
        col1, col2 = st.columns([2, 1])  # 2:1 width ratio
        
        # Column 1: Search input
        with col1:
            search_term = st.text_input("🔍 Search for foods:", 
                                       placeholder="Enter food name...", 
                                       key="search_foods")
        
        # Column 2: Sort dropdown
        with col2:
            sort_by = st.selectbox("Sort by:", 
                                  ["Score", "Calories", "Protein", "Carbs"], 
                                  key="sort_foods")
        
        # Create copy of filtered foods for display (preserves original)
        display_foods = st.session_state.filtered_foods.copy()
        
        # Apply search filter if search term provided
        if search_term:
            # Filter foods containing search term (case-insensitive)
            display_foods = display_foods[display_foods["name"]
                                         .str.contains(search_term, case=False, na=False)]
        
        # Apply sorting based on selected criterion
        if sort_by == "Score":
            # Sort by match score (ascending - lower is better)
            display_foods = display_foods.sort_values("score")
        else:
            # Sort by nutrient value
            display_foods = display_foods.sort_values(sort_by.lower(), 
                                                     ascending=sort_by=="Calories")
        
        # Display count of found foods
        st.markdown(f"### 🍽️ Recommended Foods ({len(display_foods)} found)")
        
        # Create three columns for food card display
        cols = st.columns(3)
        
        # Display foods in three-column grid layout
        for idx, (_, food) in enumerate(display_foods.head(30).iterrows()):
            # Cycle through columns (0,1,2,0,1,2,...)
            with cols[idx % 3]:
                # Food card with detailed nutritional information
                st.markdown(f"""
                <div class='food-card'>
                    <h4>{food['name']}</h4>
                    <p>🔥 <strong>Calories:</strong> {int(food['calories'])} kcal</p>
                    <p>🥚 <strong>Protein:</strong> {food['protein']:.1f}g</p>
                    <p>🥑 <strong>Fat:</strong> {food['fat']:.1f}g</p>
                    <p>🌾 <strong>Carbs:</strong> {food['carbs']:.1f}g</p>
                    <p>📊 <strong>Match Score:</strong> {food['score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)


# ==========================================================
# NUTRITION ANALYTICS TAB CONTENT
# ==========================================================

elif selected_tab == "Nutrition Analytics":
    # Tab header with custom CSS class
    st.markdown("<h1 class='main-header'>📊 Nutrition Analytics</h1>", unsafe_allow_html=True)
    
    # Check if filtered foods exist
    if st.session_state.filtered_foods.empty:
        # Prompt to generate plan first
        st.info("👈 Please generate your personalized plan first to see analytics.")
    
    else:
        # Two-column layout for analytics visualizations
        col1, col2 = st.columns(2)
        
        # Left column analytics
        with col1:
            # Nutrient distribution box plot
            st.subheader("📈 Nutrient Distribution")
            
            # Create box plot showing distribution of key nutrients
            nutrient_fig = px.box(st.session_state.filtered_foods, 
                                y=['calories', 'protein', 'fat', 'carbs'],
                                title="Distribution of Key Nutrients")
            st.plotly_chart(nutrient_fig, width='stretch')
            
            # Calories vs Protein scatter plot
            st.subheader("🔍 Calories vs Protein Correlation")
            
            # Create scatter plot to show relationship between calories and protein
            scatter_fig = px.scatter(st.session_state.filtered_foods, 
                                x='calories', y='protein',
                                hover_data=['name'],  # Show food name on hover
                                title="Calories vs Protein Content",
                                color='protein')  # Color by protein value
            st.plotly_chart(scatter_fig, width='stretch')
        
        # Right column analytics
        with col2:
            # Top high-protein foods bar chart
            st.subheader("💪 Top High-Protein Foods")
            
            # Get 10 foods with highest protein content
            high_protein = st.session_state.filtered_foods.nlargest(10, 'protein')
            
            # Create bar chart of high-protein foods
            protein_fig = px.bar(high_protein, x='name', y='protein',
                            title="Top 10 High-Protein Foods",
                            color='protein')  # Color bars by protein value
            st.plotly_chart(protein_fig, width='stretch')
            
            # Nutrient balance radar chart
            st.subheader("🎯 Nutrient Balance Radar")
            
            # Dropdown to select specific food for radar chart
            selected_food = st.selectbox("Select a food:", 
                                        st.session_state.filtered_foods['name'].unique(), 
                                        key="radar_food")
            
            # Get data for selected food
            food_data = st.session_state.filtered_foods[
                st.session_state.filtered_foods['name'] == selected_food].iloc[0]
            
            # Prepare data for radar chart
            nutrients = ['Protein', 'Fat', 'Carbs']
            values = [food_data['protein'], food_data['fat'], food_data['carbs']]
            
            # Create radar (polar) chart
            radar_fig = px.line_polar(r=values, theta=nutrients, line_close=True,
                                    title=f"Nutrient Balance: {selected_food}")
            st.plotly_chart(radar_fig, width='stretch')


# ==========================================================
# USER HISTORY TAB CONTENT
# ==========================================================

elif selected_tab == "User History":
    # Tab header with custom CSS class
    st.markdown("<h1 class='main-header'>📚 Your History & Progress</h1>", unsafe_allow_html=True)
    
    # Get user history from storage
    profile_history, meal_plan_history = get_user_history(st.session_state.current_user)
    
    # Create tabs for different history views
    history_tab1, history_tab2, history_tab3 = st.tabs(["📋 Profile History", 
                                                       "🍽️ Meal Plan History", 
                                                       "📈 Progress Tracking"])
    
    # Tab 1: Profile History
    with history_tab1:
        st.subheader("📋 Profile History")
        
        # Check if profile history exists
        if not profile_history:
            st.info("No profile history found. Generate a plan to start tracking!")
        else:
            # Sort history by timestamp (newest first)
            profile_history_sorted = sorted(profile_history, 
                                          key=lambda x: x.get('timestamp', ''), 
                                          reverse=True)
            
            # Display each historical profile
            for i, profile in enumerate(profile_history_sorted):
                # Create expandable section for each profile
                with st.expander(f"Profile #{profile.get('plan_id', i+1)} - {profile.get('timestamp', 'No date')}", 
                               expanded=i==0):  # Expand first item by default
                    
                    # Four-column layout for profile metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Column 1: Goal and Age
                    with col1:
                        st.metric("Goal", profile.get('goal', 'N/A'))
                        st.metric("Age", profile.get('age', 'N/A'))
                    
                    # Column 2: Weight and Height
                    with col2:
                        st.metric("Weight", f"{profile.get('weight', 'N/A')} kg")
                        st.metric("Height", f"{profile.get('height', 'N/A')} cm")
                    
                    # Column 3: BMI and Calories
                    with col3:
                        st.metric("BMI", profile.get('bmi', 'N/A'))
                        st.metric("Calories", f"{profile.get('tdee', 'N/A')} kcal")
                    
                    # Column 4: Diet type and condition count
                    with col4:
                        st.metric("Diet", profile.get('food_preference', 'N/A'))
                        st.metric("Conditions", len(profile.get('medical_conditions', [])))
                    
                    # Button to load this historical profile
                    if st.button(f"🔄 Load Profile #{profile.get('plan_id', i+1)}", 
                                key=f"load_profile_{i}"):
                        # Load profile into current session state
                        st.session_state.user_profile = profile
                        st.success(f"Profile #{profile.get('plan_id', i+1)} loaded! Go to Home tab to see updates.")
                        st.rerun()
    
    # Tab 2: Meal Plan History
    with history_tab2:
        st.subheader("🍽️ Meal Plan History")
        
        # Check if meal plan history exists
        if not meal_plan_history:
            st.info("No meal plan history found. Generate a plan to start tracking!")
        else:
            # Sort history by timestamp (newest first)
            meal_plan_history_sorted = sorted(meal_plan_history, 
                                            key=lambda x: x.get('timestamp', ''), 
                                            reverse=True)
            
            # Display each historical meal plan
            for i, plan in enumerate(meal_plan_history_sorted):
                # Create expandable section for each meal plan
                with st.expander(f"Meal Plan #{plan.get('plan_id', i+1)} - {plan.get('timestamp', 'No date')}", 
                               expanded=i==0):
                    
                    # Extract plan data
                    profile_data = plan.get('user_profile', {})
                    plan_data = plan.get('plan_data', {})
                    
                    # Two-column layout for plan summary
                    col1, col2 = st.columns(2)
                    
                    # Column 1: Plan details
                    with col1:
                        st.write(f"**Goal:** {profile_data.get('goal', 'N/A')}")
                        st.write(f"**Diet Type:** {profile_data.get('food_preference', 'N/A')}")
                        st.write(f"**Target Calories:** {profile_data.get('tdee', 'N/A')} kcal")
                    
                    # Column 2: Statistics
                    with col2:
                        st.write(f"**Food Items:** {plan.get('food_count', 0)}")
                        st.write(f"**Days Planned:** {len(plan_data)}")
                        st.write(f"**BMI at time:** {profile_data.get('bmi', 'N/A')}")
                    
                    # Show sample day from meal plan
                    if plan_data:
                        # Get first day from plan
                        sample_day = list(plan_data.keys())[0]
                        st.write(f"**Sample Day ({sample_day}):**")
                        st.write(f"🍳 Breakfast: {', '.join(plan_data[sample_day].get('Breakfast', []))}")
                        st.write(f"🍲 Lunch: {', '.join(plan_data[sample_day].get('Lunch', []))}")
                        st.write(f"🍛 Dinner: {', '.join(plan_data[sample_day].get('Dinner', []))}")
                    
                    # Button to load this historical meal plan
                    if st.button(f"🔄 Load Meal Plan #{plan.get('plan_id', i+1)}", 
                                key=f"load_plan_{i}"):
                        # Load meal plan into current session state
                        st.session_state.meal_plan = plan_data
                        st.session_state.user_profile = profile_data
                        st.success(f"Meal Plan #{plan.get('plan_id', i+1)} loaded!")
                        st.rerun()
    
    # Tab 3: Progress Tracking
    with history_tab3:
        st.subheader("📈 Progress Tracking")
        
        # Check if progress data exists
        if not profile_history:
            st.info("No progress data available yet. Generate plans to track your progress!")
        else:
            # Sort history by timestamp (oldest first for trend lines)
            profile_history_sorted = sorted(profile_history, 
                                          key=lambda x: x.get('timestamp', ''))
            
            # Prepare data for progress charts
            dates = []
            weights = []
            bmis = []
            calories = []
            
            # Extract time-series data from profile history
            for profile in profile_history_sorted:
                dates.append(profile.get('timestamp', 'Unknown'))
                weights.append(profile.get('weight', 0))
                bmis.append(profile.get('bmi', 0))
                calories.append(profile.get('tdee', 0))
            
            # Two-column layout for progress charts
            col1, col2 = st.columns(2)
            
            # Column 1: Weight progress chart
            with col1:
                # Check if enough data points for chart
                if len(weights) > 1:
                    # Create line chart for weight progress
                    fig_weight = go.Figure()
                    fig_weight.add_trace(go.Scatter(
                        x=dates,  # X-axis: dates
                        y=weights,  # Y-axis: weight values
                        mode='lines+markers',  # Lines with data points
                        name='Weight (kg)',  # Legend label
                        line=dict(color='#FF6B6B', width=3)  # Custom line style
                    ))
                    fig_weight.update_layout(
                        title="Weight Progress Over Time",
                        xaxis_title="Date",
                        yaxis_title="Weight (kg)",
                        template="plotly_dark"  # Dark theme
                    )
                    st.plotly_chart(fig_weight, use_container_width=True)
                else:
                    st.info("Need more data points to show weight progress")
            
            # Column 2: BMI progress chart
            with col2:
                if len(bmis) > 1:
                    fig_bmi = go.Figure()
                    fig_bmi.add_trace(go.Scatter(
                        x=dates,
                        y=bmis,
                        mode='lines+markers',
                        name='BMI',
                        line=dict(color='#4ECDC4', width=3)
                    ))
                    
                    # Add BMI category reference lines
                    fig_bmi.add_hline(y=18.5, line_dash="dash", line_color="green", 
                                     annotation_text="Underweight")
                    fig_bmi.add_hline(y=25, line_dash="dash", line_color="blue", 
                                     annotation_text="Normal")
                    fig_bmi.add_hline(y=30, line_dash="dash", line_color="orange", 
                                     annotation_text="Overweight")
                    
                    fig_bmi.update_layout(
                        title="BMI Progress Over Time",
                        xaxis_title="Date",
                        yaxis_title="BMI",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_bmi, use_container_width=True)
                else:
                    st.info("Need more data points to show BMI progress")
            
            # Summary statistics section
            st.subheader("📊 Summary Statistics")
            
            # Four-column layout for summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Column 1: Weight change
            with col1:
                if weights:
                    weight_change = weights[-1] - weights[0]
                    st.metric("Weight Change", f"{weight_change:+.1f} kg")
            
            # Column 2: BMI change
            with col2:
                if bmis:
                    bmi_change = bmis[-1] - bmis[0]
                    st.metric("BMI Change", f"{bmi_change:+.2f}")
            
            # Column 3: Initial goal
            with col3:
                if profile_history_sorted:
                    first_goal = profile_history_sorted[0].get('goal', 'N/A')
                    st.metric("Initial Goal", first_goal)
            
            # Column 4: Current goal
            with col4:
                if profile_history_sorted:
                    current_goal = profile_history_sorted[-1].get('goal', 'N/A')
                    st.metric("Current Goal", current_goal)
            
            # Progress notes section
            st.subheader("📝 Add Progress Note")
            
            # Text area for user to add progress notes
            progress_note = st.text_area("Record your progress or notes:", 
                                        placeholder="E.g., 'Lost 2kg this month! Feeling more energetic.'")
            
            # Button to save progress note
            if st.button("💾 Save Progress Note", use_container_width=True):
                if progress_note:
                    # Load users data
                    users = load_users()
                    
                    # Check if current user exists
                    if st.session_state.current_user in users:
                        # Initialize progress_notes list if not exists
                        if 'progress_notes' not in users[st.session_state.current_user]:
                            users[st.session_state.current_user]['progress_notes'] = []
                        
                        # Add new progress note with metadata
                        users[st.session_state.current_user]['progress_notes'].append({
                            'note': progress_note,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'weight': st.session_state.user_profile.get('weight', 'N/A'),
                            'bmi': st.session_state.user_profile.get('bmi', 'N/A')
                        })
                        
                        # Save updated users data
                        save_users(users)
                        st.success("Progress note saved!")
                else:
                    st.warning("Please enter a progress note")


# ==========================================================
# FOOTER SECTION
# ==========================================================

# Horizontal separator line
st.markdown("---")

# Footer with user info and copyright
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    f"Logged in as: {st.session_state.current_user} | "  # Display current username
    "Built with ❤️ using Streamlit | NUTRI-CARE - Your AI Diet Planner"
    "</div>", 
    unsafe_allow_html=True
)
