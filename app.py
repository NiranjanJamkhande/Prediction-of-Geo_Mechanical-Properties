import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Load the trained model
joblib_file = "random_forest_model.pkl"
rf = joblib.load(joblib_file)

st.set_page_config(
    page_title='Geo-Mechanical Properties Prediction',
    page_icon='✅',
    layout='wide'
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        margin-top: 0;
        padding-top: 0;
    }
    .subheader {
        font-size: 28px;
        font-weight: bold;
        color: #555;
        margin-top: 20px;
        padding-top: 0;
    }
    .content {
        margin-top: 30px;
        padding-top: 0;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">Geo-Mechanical Properties Prediction</h1>', unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload CSV")
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(st.session_state.uploaded_file)

    # Set the depth column as the index
    if 'Depth' in data.columns:
        data.set_index('Depth', inplace=True)

    # Read the other dataframe with the actual column
    actual_data = pd.read_csv("Comparing_csv.csv")

    # Add the 'actual column' from the other dataframe to input_data
    data['Actual Poisson Ratio(u)'] = pd.Series(actual_data['Actual Poisson Ratio(u)'].values, index=data.index)
    data['Actual Young Modulus(E)'] = pd.Series(actual_data['Actual Young Modulus(E)'].values, index=data.index)

    # Assuming the CSV has columns: resistivity, gamma_ray, total_porosity, bulk_density
    input_data = data[['Resistivity', 'Gamma Ray', 'Total Porosity', 'Bulk Density']]

    # Make predictions
    predictions = rf.predict(input_data)

    # Add predictions to the dataframe
    data['Predicted Poisson Ratio(u)'] = predictions[:, 0]
    data['Predicted Young Modulus(E)'] = predictions[:, 1]

    # Plot for Young Modulus(E)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data['Actual Young Modulus(E)'], mode='lines', name="Actual Young's Modulus", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=data.index, y=data['Predicted Young Modulus(E)'], mode='lines', name="Predicted Young's Modulus", line=dict(color='red')))
    fig1.update_layout(
        title="Actual vs Predicted Young's Modulus",
        xaxis_title="Depth (ft)",
        yaxis_title="Young’s Modulus (GPa)",
        legend_title="Legend",
        template="plotly_white",
        width=680,
        height=400,
        margin=dict(l=40, r=10, t=40, b=30)
    )

    # Plot for Poisson Ratio(u)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Actual Poisson Ratio(u)'], mode='lines', name='Actual Dynamic Poisson’s Ratio', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=data.index, y=data['Predicted Poisson Ratio(u)'], mode='lines', name='Predicted Dynamic Poisson’s Ratio', line=dict(color='red')))
    fig2.update_layout(
        title="Actual vs Predicted Poisson Ratio",
        xaxis_title="Depth (ft)",
        yaxis_title="Dynamic Poisson’s Ratio ϑ",
        legend_title="Legend",
        template="plotly_white",
        width=680,
        height=400,
        margin=dict(l=40, r=10, t=40, b=30)
    )

    # Create columns for plots
    col1, col2 = st.columns([1, 1])

    # Render the Young Modulus plot in the left column
    col1.plotly_chart(fig1, use_container_width=True)

    # Render the Poisson Ratio plot in the right column
    col2.plotly_chart(fig2, use_container_width=True)

    # Limit the number of records displayed
    #limited_data = data.head(50)

    # Main content area
    st.markdown('<h2 class="subheader">Predicted Geo-Mechanical Properties</h2>', unsafe_allow_html=True)

    # Display the dataframe with predictions
    def highlight_predictions(column):
        if column.name in ['Actual Poisson Ratio(u)', 'Predicted Poisson Ratio(u)']:
            return ['background-color: lightblue; font-weight: bold; color: #333' for _ in column]
        elif column.name in ['Actual Young Modulus(E)', 'Predicted Young Modulus(E)']:
            return ['background-color: lightcoral; font-weight: bold; color: #333' for _ in column]
        else:
            return ['font-weight: bold; color: #333' for _ in column]

    st.dataframe(
        data.style.apply(highlight_predictions, axis=0).set_table_styles(
            [
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', '#fff'), ('font-weight', 'bold'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'center'), ('font-weight', 'bold'), ('color', '#333')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#f0f8ff')]}
            ]
        ),
        use_container_width=True
    )

    # Provide download link for the results
    csv = data.to_csv(index=True)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='predicted_geo_mechanical_properties.csv',
        mime='text/csv'
    )
import sklearn
