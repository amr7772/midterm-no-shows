"""Gradio interface for the Maple Physio no-show prediction project."""

import gradio as gr


def predict_no_show(
    age,
    lead_days,
    prior_no_shows,
    hour,
    weekday,
    booking,
    insurance,
    treatment,
    first_visit,
    distance_km,
):
    """Temporary response until the prediction classes are available."""
    return (
        "Model connection pending",
        "The prediction classes are still being integrated.",
    )


with gr.Blocks(title="Maple Physio No-Show Prediction") as demo:
    gr.Markdown("# Maple Physio – No-Show Prediction")
    gr.Markdown(
        "Enter appointment information to estimate the risk "
        "of a patient missing their visit."
    )

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Patient Age", value=35)
            lead_days = gr.Number(label="Days Until Appointment", value=7)
            prior_no_shows = gr.Number(label="Previous Missed Appointments", value=0)
            hour = gr.Slider(
                minimum=8, maximum=17, step=1, value=10, label="Appointment Hour"
            )
            weekday = gr.Dropdown(
                choices=["mon", "tue", "wed", "thu", "fri"],
                value="mon",
                label="Weekday",
            )
            booking = gr.Dropdown(
                choices=["online", "phone", "front_desk"],
                value="online",
                label="Booking Method",
            )

        with gr.Column():
            insurance = gr.Dropdown(
                choices=["private", "provincial", "self_pay"],
                value="private",
                label="Insurance",
            )
            treatment = gr.Dropdown(
                choices=["physio", "massage", "acupuncture"],
                value="physio",
                label="Treatment",
            )
            first_visit = gr.Radio(
                choices=[0, 1],
                value=0,
                label="First Visit? (0 = No, 1 = Yes)",
            )
            distance_km = gr.Number(
                label="Distance From Clinic (km)",
                value=10,
            )

    predict_button = gr.Button("Predict No-Show", variant="primary")

    probability_output = gr.Textbox(
        label="No-Show Probability",
        interactive=False,
    )

    recommendation_output = gr.Textbox(
        label="Recommendation",
        interactive=False,
    )

    predict_button.click(
        fn=predict_no_show,
        inputs=[
            age,
            lead_days,
            prior_no_shows,
            hour,
            weekday,
            booking,
            insurance,
            treatment,
            first_visit,
            distance_km,
        ],
        outputs=[probability_output, recommendation_output],
    )


if __name__ == "__main__":
    demo.launch()
