import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine
import pymysql


# 🌟 Set Page Config
st.set_page_config(page_title="🌾 Agriculture Prediction", layout="wide")

# 🌟 Sidebar Styling with Custom CSS
st.markdown(
    """
    <style>
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #004d99, #4CAF50);
            color: white;
            padding: 20px;
        }

        /* Sidebar Title */
        [data-testid="stSidebar"] h1 {
            color: #FFD700;  /* Gold Color */
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }

        /* Sidebar Radio Button Styling */
        div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            background: #004d99;
            padding: 10px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
            margin-bottom: 10px;
        }

        /* Hover Effect */
        div[role="radiogroup"] label:hover {
            background: #FFA500;  /* Orange */
            transform: scale(1.05);
            cursor: pointer;
        }

        /* Selected Option */
        div[role="radiogroup"] input:checked + div {
            background: #FFD700; /* Gold */
            color: black;
            font-weight: bold;
            box-shadow: 0px 0px 10px rgba(255, 215, 0, 0.8);
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Select One")
# 🌟 Sidebar Navigation with Icons
page = st.sidebar.radio("📌 **Navigation**", 
                        ["📊 Dashboard", "🔮 Predictions"])


# 🚀 **PAGE 1: Dashboard**
if page == "📊 Dashboard":
    st.markdown('<h1 style="text-align:center; color:white;">🌾 Agriculture Data Dashboard</h1>', unsafe_allow_html=True)

    # App background style
    st.markdown(
        """
        <style>
            .stApp {
                background: url("https://source.unsplash.com/1600x900/?nature,green") no-repeat center center fixed;
                background-size: cover;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid white;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
   # MySQL Database Connection Settings
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "vijay45"
    DB_NAME = "agriculture"
    DB_PORT = 3306

    # Create MySQL Database Connection
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
   
    # Query to fetch data
    query = "SELECT * FROM raw"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    df["year"] = df["year"].astype(str)  # Convert 'year' to string for better display

    # Function to highlight max values
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    
    # Sidebar Filters
    st.sidebar.title("💻 Analysis")
    st.sidebar.header("Filters for Production")

    # Set default values
    # Default values
    default_country = "India"
    default_items = ["Apples", "Barley"]

    # Get unique values for area and items
    area_list = sorted(df["area"].unique())
    item_list = df["item"].unique()

    # Ensure default values exist before setting them
    default_area_index = area_list.index(default_country) if default_country in area_list else 0
    default_selected_items = [item for item in default_items if item in item_list]

    # Sidebar selection with safe defaults
    selected_area = st.sidebar.selectbox("Select Area", area_list, index=default_area_index)
    selected_items = st.sidebar.multiselect("Select Items", item_list, default=default_selected_items)


    
    #selected_area = st.sidebar.selectbox("Select Area", sorted(df["area"].unique(), reverse=True))
    #selected_items = st.sidebar.multiselect("Select Items", df["item"].unique())
    
    filter_df = df[df["area"] == selected_area]
    if selected_items:
        filter_df = filter_df[filter_df["item"].isin(selected_items)]
    
    st.header(f"Showing Data for Area : {selected_area}")
    styled_df = filter_df.style.apply(highlight_max, subset=["production"]) \
        .set_properties(**{"background-color": "#EAF2F8", "color": "black", "border": "1px solid black"})
    
    st.dataframe(styled_df)

    # Side-by-side plots
    col1, col2 = st.columns(2)
    
    with col1:
     st.write("### 🥧 Production Distribution Over Time")
     production_data = filter_df.groupby("item")["production"].sum()
    
     fig1, ax1 = plt.subplots(figsize=(6, 4))
     ax1.pie(production_data, labels=production_data.index, autopct="%1.1f%%", startangle=140, 
            wedgeprops={"edgecolor": "white"})  # Remove borders
     ax1.set_title("Production Share by Item", fontsize=12, color="yellow")
    
     fig1.patch.set_facecolor("#1C2833")  # Match Streamlit's dark mode background
     ax1.set_facecolor("#FFFFFF")  # Match background color

     st.pyplot(fig1)

    with col2:
     st.write("### 📊 Top 5 Producing Items")
     top_items = filter_df.groupby("item")["production"].sum().nlargest(5).reset_index()
    
     fig2, ax2 = plt.subplots(figsize=(6, 6))
     sns.barplot(x="item", y="production", data=top_items, ax=ax2, palette="coolwarm")

     ax2.set_title("Top 5 Production Items", fontsize=12, color="white")
     ax2.set_xlabel("Item", fontsize=10, color="white")
     ax2.set_ylabel("Total Production", fontsize=10, color="white")
     ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, color="white")
     ax2.set_yticklabels(ax2.get_yticks(), color="white")

     fig2.patch.set_facecolor("#1C2833")  # Match Streamlit's dark mode background
     ax2.set_facecolor("#1C2833")  # Match background color
     ax2.spines["top"].set_visible(False)  # Remove top border
     ax2.spines["right"].set_visible(False)  # Remove right border
     ax2.spines["left"].set_visible(False)  # Remove left border
     ax2.spines["bottom"].set_visible(False)  # Remove bottom border
     ax2.grid(False)  # Remove gridlines

     st.pyplot(fig2)

    
    # Year-wise Harvested Analysis


    st.sidebar.header("Year-wise Harvested")
    selected_area_year = st.sidebar.multiselect("Select Country", df["area"].unique(), key="year_selection")
    filtered_df = df[df["area"].isin(selected_area_year)]
    
    if not filtered_df.empty:
        selected_years = st.sidebar.multiselect("Select Year", filtered_df["year"].unique(), key="country_selection")
        filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
    
    if not filtered_df.empty:
        selected_item = st.sidebar.multiselect("Select Item", filtered_df["item"].unique(), key="item_selection")
        filtered_df = filtered_df[filtered_df["item"].isin(selected_item)]
    
    # Display Filtered Data
    st.write("### Area Harvested Over Years")
    styled_filtered_df = filtered_df.style.apply(highlight_max, subset=["area_harvested"]) \
        .set_properties(**{"background-color": "#FAD7A0", "color": "black", "border": "1px solid black"})
    
    st.dataframe(styled_filtered_df)

    if not filtered_df.empty:
        st.write("### 🌊 Harvested Area Over Time (Stream Graph)")
        filtered_df["year"] = pd.to_numeric(filtered_df["year"], errors="coerce")
        filtered_df = filtered_df.dropna(subset=["year"])
        filtered_df = filtered_df.sort_values(by="year")
        pivot_df = filtered_df.pivot_table(index="year", columns="item", values="area_harvested", aggfunc="sum")
    
        if not pivot_df.empty:
         fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1C2833")  # Match background color

         pivot_df.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)

         ax.set_title("Harvested Area Over Years", fontsize=14, fontweight='bold', color="#ECF0F1")
         ax.set_xlabel("Year", fontsize=12, color="#FFA500")
         ax.set_ylabel("Total Area Harvested", fontsize=12, color="#FFA500")

         ax.legend(title="Items", loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
         ax.set_facecolor("#ECF0F1")  # Match the background color
         plt.xticks(rotation=45, color="#ECF0F1")
         plt.yticks(color="#FFA500")
    
    # Remove all borders to match the background
         for spine in ax.spines.values():
          spine.set_visible(False)

         st.pyplot(fig)
        else:
         st.warning("Select the data to filter.")


    else:
        st.warning(" Select the Data in Yearly-wise-Harvested section to filter.")


     
      ### Page 2





elif page == "🔮 Predictions":

    # Set Streamlit page config
#st.set_page_config(page_title="🌾 Agriculture Production Prediction", layout="wide")

# App background style
    st.markdown(
    """
    <style>
        .stApp {
            background: url("https://source.unsplash.com/1600x900/?nature,green") no-repeat center center fixed;
            background-size: cover;
        }
        .header {
            text-align: center;
            color: Orange;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 24px;
            color: #FFA500;
            font-weight: bold;
        }
        .prediction-box {
            background-color: #2E2E2E;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .year {
            font-size: 22px;
            color: #FFA500;
        }
        .production {
            font-size: 24px;
            color: #00FF00;
        }
    </style>
    """,
    unsafe_allow_html=True
)

    # Header
    st.markdown('<div class="header">🌾 Agriculture Production Prediction</div>', unsafe_allow_html=True)



    # Load dataset
df = pd.read_csv(r"C:\Users\aswin\3D Objects\guvi_project\project-3 (crops_agriculture)\Capped.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

def highlight_cells(val):
    """Highlight specific cells based on value conditions"""
    color = "background-color: #1E90FF" 
    return color





# User input for filtering data
filter1 = st.sidebar.text_input("Enter country or item to filter data:", "")

if filter1:
    filtered_data = df[(df["area"] == filter1) | (df["item"] == filter1)]
    
    if filtered_data.empty:
        st.write("No data found for the entered country or item.")
    else:
        st.write(f"Filtered Data for: {filter1}")
        st.dataframe(filtered_data.style.applymap(highlight_cells))
        #st.dataframe(filtered_data)

        # Further filtering
        filter2 = st.sidebar.text_input(f"Enter country or item to filter from the results (same as '{filter1}'):", "")

        if filter2:
            final_filtered_data = filtered_data[(filtered_data["area"] == filter2) | (filtered_data["item"] == filter2)]
            
            if final_filtered_data.empty:
                st.write("No data found for the second filter.")
            else:
                st.write(f"Final Filtered Data for: {filter2}")
                st.dataframe(final_filtered_data.style.applymap(highlight_cells))
                #st.dataframe(final_filtered_data)

                # Prepare data
                df_capped = final_filtered_data.dropna(subset=['production', 'year', 'area_harvested', 'yield'])
                X = df_capped.drop(columns=["production"])
                Y = df_capped["production"]

                # One-hot encoding for categorical variables
                X = pd.get_dummies(X, columns=["area", "item"], drop_first=True)

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Model Evaluation
                y_pred = model.predict(X_train)
                mae = mean_absolute_error(y_train, y_pred)
                mse = mean_squared_error(y_train, y_pred)
                r2 = r2_score(y_train, y_pred)

                

                 # Display Model Evaluation Metrics with Colors
                st.markdown("""
                <div style="background-color:#2E2E2E; padding:15px; border-radius:10px; text-align:center;">
                <h3 style="color:#FFA500;">📊 Model Evaluation Metrics</h3>
                <p style="color:white; font-size:18px;">
                <span style="color:#00FF00;">✔ Mean Absolute Error (MAE):</span> <b style="color:#FFD700;">{:.2f}</b>
                <br>
                <span style="color:#00FF00;">✔ Mean Squared Error (MSE):</span> <b style="color:#FFD700;">{:.2f}</b>
                <br>
                <span style="color:#00FF00;">✔ R-squared Score:</span> <b style="color:#FFD700;">{:.2f}</b>
                </p>
                </div>
                """.format(mae, mse, r2), unsafe_allow_html=True)


                # === Data Visualizations ===
                st.markdown("### Data Visualizations Before Prediction")

                # Correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df_capped[['production', 'area_harvested', 'yield']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                st.pyplot(fig)

                # Distribution of 'production'
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_capped['production'], kde=True, color='yellow', bins=20, ax=ax)
                ax.set_title('Distribution of Production')
                st.pyplot(fig)

                # Scatter plot: 'area_harvested' vs 'yield'
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_capped, x='area_harvested', y='yield', color='red', ax=ax)
                ax.set_title('Area Harvested vs Yield')
                st.pyplot(fig)

                # Predict Future Production for 2024 & 2025 with different growth rates
                future_years = [df_capped["year"].max() + i for i in range(1, 3)]
                area_growth, yield_growth = [], []
                prev_area = df_capped["area_harvested"].iloc[-1]
                prev_yield = df_capped["yield"].iloc[-1]

                # Generate different growth rates for each year
                growth_rates = [0.15 + np.random.uniform(0.01, 0.03),  # 2024 Growth
                                0.15 + np.random.uniform(0.04, 0.07)]  # 2025 Growth

                for i in range(2):
                    prev_area *= (1 + growth_rates[i])
                    prev_yield *= (1 + growth_rates[i])
                    area_growth.append(prev_area)
                    yield_growth.append(prev_yield)

                # Create future years DataFrame
                next_years = pd.DataFrame({"year": future_years, "area_harvested": area_growth, "yield": yield_growth})
                for col in X.columns:
                    if col not in next_years:
                        next_years[col] = X_train[col].mode()[0]
                next_years = next_years[X.columns]

                # Predict future production
                future_preds = model.predict(next_years)

                # Extract predictions for 2024 and 2025
                pred_2024, pred_2025 = future_preds
                production_diff = pred_2025 - pred_2024

                # Display Future Predictions
                st.markdown("### Future Predictions for 2024 and 2025:")
                for year, preds in zip(future_years, future_preds):
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <span class="year">📅 Year: {year}</span><br>
                            <span class="production">🌾 Predicted Production: {preds:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Show difference between 2024 & 2025 predictions
                st.markdown(
                    f"""
                    <div class="prediction-box" style="background-color: #FF5733;">
                        <span class="year">📊 Difference in Production (2025 - 2024):</span><br>
                        <span class="production">🔼 {production_diff:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Time series plot with future predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                df_capped.groupby('year')['production'].sum().plot(kind='line', color='blue', marker='o', ax=ax)
                pd.Series(future_preds, index=future_years).plot(kind='line', color='orange', marker='o', ax=ax, linestyle='--', linewidth=2)
                ax.set_title('Production Over Time with Future Predictions')
                st.pyplot(fig)
