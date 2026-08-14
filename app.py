import os
import gradio as gr
import joblib
import pandas as pd

# Load trained model
model = joblib.load("house_price_prediction_model.pkl")


def predict_price(area, bedrooms, bathrooms, floors,
                  year_built, location, condition, garage):

    try:
        # Create DataFrame with EXACT feature names
        input_data = pd.DataFrame([{
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "YearBuilt": year_built,
            "Location": location,
            "Condition": condition,
            "Garage": garage
        }])

        prediction = model.predict(input_data)[0]

        return f"Predicted House Price: ₹{prediction:,.2f}"

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio Interface
with gr.Blocks(title="House Price Predictor") as demo:

    gr.Markdown(
        """
        # 🏠 House Price Predictor
        ### Predict the estimated price of a house using Machine Learning
        """
    )

    with gr.Row():

        with gr.Column():
            area = gr.Number(
                label="Area",
                value=1500,
                minimum=1
            )

            bedrooms = gr.Number(
                label="Bedrooms",
                value=3,
                minimum=1,
                precision=0
            )

            bathrooms = gr.Number(
                label="Bathrooms",
                value=2,
                minimum=1
            )

            floors = gr.Number(
                label="Floors",
                value=2,
                minimum=1
            )

        with gr.Column():
            year_built = gr.Number(
                label="Year Built",
                value=2015,
                minimum=1800,
                maximum=2026,
                precision=0
            )

            location = gr.Textbox(
                label="Location",
                placeholder="Enter location"
            )

            condition = gr.Textbox(
                label="Condition",
                placeholder="e.g. Good"
            )

            garage = gr.Number(
                label="Garage",
                value=1,
                minimum=0,
                precision=0
            )

    predict_button = gr.Button(
        "Predict House Price",
        variant="primary"
    )

    result = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    predict_button.click(
        fn=predict_price,
        inputs=[
            area,
            bedrooms,
            bathrooms,
            floors,
            year_built,
            location,
            condition,
            garage
        ],
        outputs=result
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
