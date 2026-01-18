import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import advanced models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate synthetic medical cost data"""
    np.random.seed(42)
    
    ages = np.random.randint(18, 65, n_samples)
    bmi = np.random.normal(30, 6, n_samples)
    bmi = np.clip(bmi, 15, 50)
    children = np.random.poisson(1, n_samples)
    children = np.clip(children, 0, 5)
    smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    sex = np.random.choice(['male', 'female'], n_samples)
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    
    base_cost = 3000
    age_effect = ages * 100
    bmi_effect = (bmi - 25) * 50
    smoker_effect = np.where(smoker == 'yes', 20000, 0)
    children_effect = children * 500
    
    charges = (base_cost + age_effect + bmi_effect + 
               smoker_effect + children_effect + 
               np.random.normal(0, 3000, n_samples))
    charges = np.maximum(charges, 1000)
    
    df = pd.DataFrame({
        'age': ages,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges
    })
    
    return df

def preprocess_data(df):
    """Preprocess and engineer features"""
    df_processed = df.copy()
    
    # Feature engineering
    df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
    df_processed['smoker_bmi'] = (df_processed['smoker'] == 'yes').astype(int) * df_processed['bmi']
    df_processed['age_squared'] = df_processed['age'] ** 2
    df_processed['bmi_category'] = pd.cut(df_processed['bmi'], 
                                           bins=[0, 18.5, 25, 30, 100], 
                                           labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])
    df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])
    
    # One-hot encode
    df_processed = pd.get_dummies(df_processed, columns=['region'], prefix='region', drop_first=True)
    df_processed = pd.get_dummies(df_processed, columns=['bmi_category'], prefix='bmi_cat', drop_first=True)
    
    # Drop original categorical columns
    df_processed = df_processed.drop(['sex', 'smoker'], axis=1)
    
    return df_processed

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'predictions': y_pred
        }
    
    return results

def predict_single_cost(model, scaler, input_data, feature_names):
    """Predict cost for a single patient"""
    # Create DataFrame with all features
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Cost Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict healthcare costs using machine learning")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Data", "Upload CSV File"]
    )
    
    # Load data
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
        else:
            st.sidebar.info("👆 Please upload a CSV file")
            df = generate_sample_data(1000)
            st.sidebar.warning("Using sample data for now...")
    else:
        df = generate_sample_data(1000)
        st.sidebar.success("✅ Sample data loaded!")
    
    # Model selection
    st.sidebar.markdown("---")
    available_models = ['Random Forest', 'Gradient Boosting']
    if XGBOOST_AVAILABLE:
        available_models.append('XGBoost')
    if LIGHTGBM_AVAILABLE:
        available_models.append('LightGBM')
    
    selected_model = st.sidebar.selectbox(
        "Select Model:",
        available_models,
        index=len(available_models)-1 if len(available_models) > 2 else 0
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Performance", 
        "🎯 Make Prediction",
        "📈 Feature Insights"
    ])
    
    # Preprocess data
    df_processed = preprocess_data(df)
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    with st.spinner('🔄 Training models...'):
        results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # ========================================================================
    # TAB 1: DATA OVERVIEW
    # ========================================================================
    with tab1:
        st.markdown('<p class="sub-header">📋 Dataset Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Avg Cost", f"${df['charges'].mean():,.2f}")
        with col4:
            st.metric("Max Cost", f"${df['charges'].max():,.2f}")
        
        st.markdown("---")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Data Types & Missing Values")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Missing': df.isnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Charges distribution
            fig = px.histogram(df, x='charges', nbins=50, 
                             title='Distribution of Medical Charges',
                             labels={'charges': 'Charges ($)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Smoker vs Charges
            avg_by_smoker = df.groupby('smoker')['charges'].mean().reset_index()
            fig = px.bar(avg_by_smoker, x='smoker', y='charges',
                        title='Average Charges by Smoking Status',
                        labels={'charges': 'Average Charges ($)', 'smoker': 'Smoker'},
                        color='smoker')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age vs Charges scatter
            fig = px.scatter(df, x='age', y='charges', color='smoker',
                           title='Age vs Medical Charges',
                           labels={'charges': 'Charges ($)', 'age': 'Age'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # BMI vs Charges scatter
            fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                           title='BMI vs Medical Charges',
                           labels={'charges': 'Charges ($)', 'bmi': 'BMI'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 2: MODEL PERFORMANCE
    # ========================================================================
    with tab2:
        st.markdown('<p class="sub-header">🎯 Model Performance Comparison</p>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
        best_result = results[best_model_name]
        
        with col1:
            st.metric(
                "Best Model",
                best_model_name,
                f"R² = {best_result['r2']:.4f}"
            )
        with col2:
            st.metric(
                "RMSE",
                f"${best_result['rmse']:,.2f}",
                delta=None
            )
        with col3:
            st.metric(
                "MAE",
                f"${best_result['mae']:,.2f}",
                delta=None
            )
        
        st.markdown("---")
        
        # Model comparison table
        st.subheader("📊 All Models Comparison")
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': result['r2'],
                'RMSE ($)': result['rmse'],
                'MAE ($)': result['mae']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('R² Score', ascending=False)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comparison_df, x='Model', y='R² Score',
                        title='Model Comparison - R² Score',
                        color='R² Score',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='RMSE ($)',
                        title='Model Comparison - RMSE',
                        color='RMSE ($)',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted
        st.markdown("---")
        st.subheader("🎯 Actual vs Predicted (Selected Model)")
        
        selected_result = results[selected_model]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=selected_result['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.update_layout(
            title=f'{selected_model} - Actual vs Predicted',
            xaxis_title='Actual Charges ($)',
            yaxis_title='Predicted Charges ($)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3: MAKE PREDICTION
    # ========================================================================
    with tab3:
        st.markdown('<p class="sub-header">🎯 Predict Medical Costs</p>', unsafe_allow_html=True)
        st.markdown("Enter patient information to predict medical costs:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 64, 30)
            bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        
        with col2:
            sex = st.selectbox("Sex", ['male', 'female'])
            smoker = st.selectbox("Smoker", ['no', 'yes'])
        
        with col3:
            children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
            region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
        
        if st.button("🔮 Predict Cost", type="primary"):
            # Create input data
            input_data = {
                'age': age,
                'bmi': bmi,
                'children': children,
                'sex_encoded': 1 if sex == 'male' else 0,
                'smoker_encoded': 1 if smoker == 'yes' else 0,
                'age_bmi': age * bmi,
                'smoker_bmi': (1 if smoker == 'yes' else 0) * bmi,
                'age_squared': age ** 2
            }
            
            # Add region dummies
            for r in ['northwest', 'southeast', 'southwest']:
                input_data[f'region_{r}'] = 1 if region == r else 0
            
            # Add BMI category dummies
            if bmi < 18.5:
                bmi_cat = 'underweight'
            elif bmi < 25:
                bmi_cat = 'normal'
            elif bmi < 30:
                bmi_cat = 'overweight'
            else:
                bmi_cat = 'obese'
            
            for cat in ['normal', 'overweight', 'obese']:
                input_data[f'bmi_cat_{cat}'] = 1 if bmi_cat == cat else 0
            
            # Make prediction
            model = results[selected_model]['model']
            prediction = predict_single_cost(model, scaler, input_data, X.columns.tolist())
            
            # Display result
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Annual Cost", f"${prediction:,.2f}")
            
            with col2:
                monthly_cost = prediction / 12
                st.metric("Monthly Cost", f"${monthly_cost:,.2f}")
            
            with col3:
                avg_cost = df['charges'].mean()
                diff = ((prediction - avg_cost) / avg_cost) * 100
                st.metric("vs Average", f"{diff:+.1f}%")
            
            # Risk factors
            st.markdown("---")
            st.subheader("📋 Risk Factor Analysis")
            
            risk_factors = []
            if smoker == 'yes':
                risk_factors.append("🚬 Smoking significantly increases costs")
            if bmi > 30:
                risk_factors.append("⚠️ Obesity (BMI > 30) is a major cost driver")
            if age > 50:
                risk_factors.append("👴 Age over 50 increases healthcare costs")
            if bmi < 18.5:
                risk_factors.append("⚠️ Underweight (BMI < 18.5) may indicate health issues")
            
            if risk_factors:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.info("✅ No major risk factors identified")
    
    # ========================================================================
    # TAB 4: FEATURE INSIGHTS
    # ========================================================================
    with tab4:
        st.markdown('<p class="sub-header">📈 Feature Importance Analysis</p>', unsafe_allow_html=True)
        
        # Get feature importance from the selected model
        model = results[selected_model]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Top 10 features
            st.subheader(f"🏆 Top 10 Features - {selected_model}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(importance_df.head(10), 
                           x='Importance', 
                           y='Feature',
                           orientation='h',
                           title='Feature Importance Ranking',
                           color='Importance',
                           color_continuous_scale='Viridis')
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Top 5 Cost Drivers:**")
                for idx, row in importance_df.head(5).iterrows():
                    st.metric(
                        row['Feature'],
                        f"{row['Importance']:.4f}"
                    )
            
            # All features table
            st.markdown("---")
            st.subheader("📊 All Features")
            st.dataframe(importance_df, use_container_width=True)
            
            # Key insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Model Explanation:**
                - The {selected_model} model explains **{results[selected_model]['r2']*100:.2f}%** of variance in medical costs
                - Average prediction error: **${results[selected_model]['rmse']:,.2f}**
                - The model is {'highly' if results[selected_model]['r2'] > 0.85 else 'moderately'} accurate
                """)
            
            with col2:
                top_feature = importance_df.iloc[0]['Feature']
                top_importance = importance_df.iloc[0]['Importance']
                
                st.success(f"""
                **Top Cost Driver:**
                - **{top_feature}** has the highest importance ({top_importance:.4f})
                - This feature alone accounts for a significant portion of cost variation
                - Focus interventions on this area for maximum impact
                """)
        else:
            st.warning("Feature importance not available for this model type")

if __name__ == "__main__":
    main()