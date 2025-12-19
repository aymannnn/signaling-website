import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Streamlit UI configuration
# -----------------------------
st.set_page_config(
    page_title="Optimal Signaling Predictor",
    layout="centered",
)

st.title("Signal Prediction Simulator")

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_model():
    # Expects the model file to be in the same directory you run Streamlit from.
    return joblib.load("knn_scaled_model.joblib")

knn_loaded = load_model()

# these should be the bounds that the streamlit user
# can slide to select from
feature_bounds = {
    'n_programs': (10, 800),
    'n_positions': (10, 10000),
    'n_applicants': (20, 18000),
    'interviews_per_spot': (5, 20),
    'max_applications': (5, 60)
}

# this should be the default slider values
feature_defaults = {
    'n_programs': 300,
    'n_positions': 1500,
    'n_applicants': 3500,
    'interviews_per_spot': 10,
    'max_applications': 40
}

def check_inputs(inputs):
    # do not allow the user to select more programs then positions
    # if they do, do NOT allow them to click "Run Signal Prediction"
    if inputs['n_programs'] > inputs['n_positions']:
        raise ValueError(
            "Number of programs cannot exceed number of positions. (Each program must have at least one position.)")
    if inputs['n_programs'] < inputs['max_applications']:
        raise ValueError(
            "Number of programs cannot be less than max applications per applicant. (Applicants cannot apply to more programs than exist.)")


# run this function once the user clicks "Run Signal Prediction"
def process_streamlit_inputs(inputs):
    # once the user selects their inputs, they have to
    # click the "Run Signal Prediction" button
    spots_per_program = inputs['n_positions'] // inputs['n_programs']
    simulated_positions = inputs['n_programs'] * (spots_per_program)
    # print "simulated positions to user"
    print(f"Simulated Positions: {simulated_positions}")


    predictor_X = pd.DataFrame()
    signals = []
    # at least 0 to 4 signals, you always have to apply to at least
    # 5 "max applications" so there is always room for this
    # note that if max applications is 5 then subtract 3 gets to 2, and we want
    # in this scenario to test 4
    for signal in range(0, max(4, min(40, inputs['max_applications'] - 3))):
        signals.append(signal)
        predictor_set = pd.DataFrame([{
            'n_programs': inputs['n_programs'],
            'simulated_positions': simulated_positions,
            'n_applicants': inputs['n_applicants'],
            'interviews_per_spot': inputs['interviews_per_spot'],
            'max_applications': inputs['max_applications'],
            'signal_value': signal
        }])
        if predictor_X.empty:
            predictor_X = predictor_set
        else:
            predictor_X = pd.concat(
                [predictor_X, predictor_set], ignore_index=True)

    processed_inputs = {}
    processed_inputs['predictor_X'] = predictor_X
    processed_inputs['signal_range'] = signals
    processed_inputs['simulated_positions'] = simulated_positions  # for Streamlit display

    return processed_inputs

# once you get the dataframes, predict and graph and circle minimum

def run_simulation(processed_inputs):
    x_axis = processed_inputs['signal_range']
    predictor_X = processed_inputs['predictor_X']
    predictor_X.sort_values(by=['signal_value'], inplace=True)
    y_pred = knn_loaded.predict(predictor_X)
    
    # now graph the X axis versus the y axis
    # ideally the graph will circle the minimum
    
    ## GRAPH HERE, MAKE IT PRETTY, CIRCLE THE MINIMUM IF YOU CAN
    
    # Graph title: Predicted Reviews per Program vs. Signal Value
    
    ## Also print for the user ABOVE the graph:
    # The optimal signal value is X.
    
    return y_pred


def _make_inputs_ui():
    st.subheader("Inputs")

    n_programs = st.slider(
        "Number of programs",
        min_value=feature_bounds["n_programs"][0],
        max_value=feature_bounds["n_programs"][1],
        value=feature_defaults["n_programs"],
        step=1,
    )

    n_positions = st.slider(
        "Number of positions",
        min_value=feature_bounds["n_positions"][0],
        max_value=feature_bounds["n_positions"][1],
        value=feature_defaults["n_positions"],
        step=1,
    )

    n_applicants = st.slider(
        "Number of applicants",
        min_value=feature_bounds["n_applicants"][0],
        max_value=feature_bounds["n_applicants"][1],
        value=feature_defaults["n_applicants"],
        step=1,
    )

    interviews_per_spot = st.slider(
        "Interviews per spot",
        min_value=feature_bounds["interviews_per_spot"][0],
        max_value=feature_bounds["interviews_per_spot"][1],
        value=feature_defaults["interviews_per_spot"],
        step=1,
    )

    max_applications = st.slider(
        "Max applications per applicant",
        min_value=feature_bounds["max_applications"][0],
        max_value=feature_bounds["max_applications"][1],
        value=feature_defaults["max_applications"],
        step=1,
    )

    return {
        "n_programs": int(n_programs),
        "n_positions": int(n_positions),
        "n_applicants": int(n_applicants),
        "interviews_per_spot": int(interviews_per_spot),
        "max_applications": int(max_applications),
    }


def _plot_predictions(signal_values, y_pred, optimal_signal, optimal_pred):
    fig, ax = plt.subplots()
    ax.plot(signal_values, y_pred, marker="o")
    ax.scatter([optimal_signal], [optimal_pred], s=200, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_title("Predicted Reviews per Program vs. Signal Value")
    ax.set_xlabel("Signal Value")
    ax.set_ylabel("Predicted Reviews per Program")
    ax.grid(True, alpha=0.3)
    return fig


def main():
    inputs = _make_inputs_ui()

    # Validate (and disable the button if invalid)
    error_msg = None
    try:
        check_inputs(inputs)
    except ValueError as e:
        error_msg = str(e)

    if error_msg:
        st.error(error_msg)

    run_clicked = st.button("Run Signal Prediction", disabled=bool(error_msg))

    if run_clicked:
        with st.spinner("Running simulation..."):
            processed = process_streamlit_inputs(inputs)
            y_pred = run_simulation(processed)

            # Display simulated positions (requested in comments)
            st.markdown(f"**Simulated Positions:** {processed['simulated_positions']:,}")

            # Find and display the optimal signal value (requested in comments)
            y_pred_arr = np.array(y_pred)
            min_idx = int(np.argmin(y_pred_arr))
            signal_values = list(processed["signal_range"])
            optimal_signal = signal_values[min_idx]
            optimal_pred = float(y_pred_arr[min_idx])

            st.markdown(f"**The optimal signal value is {optimal_signal}.**")

            # Plot and circle the minimum
            fig = _plot_predictions(signal_values, y_pred_arr, optimal_signal, optimal_pred)
            st.pyplot(fig, clear_figure=True)

            # Optional: show the underlying predictions
            with st.expander("Show prediction table"):
                results_df = pd.DataFrame(
                    {"signal_value": signal_values, "predicted_reviews_per_program": y_pred_arr}
                )
                st.dataframe(results_df, use_container_width=True)


if __name__ == "__main__":
    main()
