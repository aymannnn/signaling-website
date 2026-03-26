import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import os

st.set_page_config(layout="wide")
st.title("GME Signal Graphing")

# Configuration
ANALYSES_TO_GRAPH = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]

METRIC_NAMES = {
    'p_int_given_signal_mean': 'Interview | Signal',
    'p_int_given_nosignal_mean': 'Interview | No Signal',
    'pct_matches_from_signal_mean': 'Match from Signal',
    'pct_match_from_nosignal_mean': 'Match from No Signal',
    'unfilled_positions_mean': 'Unfilled Positions',
    'reviews_per_program_mean': 'Application Reviews',
    'expect_int_per_signal_mean': 'Expected Interviews / Signal'
}

PANEL_TITLES = [
    "Panel A: Interview & Match Rates (%)",
    "Panel B: Unfilled Positions",
    "Panel C: Workload vs. Expected Interviews",
    "Panel D: Decile Match Heatmap (Base Analysis) at Signal = {}"
]

Legends = {
    'A': '''Visualization of the probability of an interview given a signal, probability of an interview without signaling, percentage of matches from signals, and percentage of matches from no signals.''',
    'B': '''Visualization of unfilled program positions, which is equal to the number of applicants who did not that match that could have filled a spot.''',
    'C': '''Visualization of the balance between program review burden and expected number of interviews from signals. Non-signals are not demonstrated as the probability of an interview and match without signaling is close to 0.''',
    'D': '''Decile matching post-match, calculated at a specified signal value. Panels A-C are graphed with means and 95/% confidence intervals.''',
    'Residual': '''Visualization of the trade-offs in minimizing application reviews versus maximizing expected interviews (signaled). Panel A shows the absolute difference in the signal values at which the local optimum occurs. Panel B shows the relative increase in reviews at the signal value where maximum interviews occur. Likewise, panel C shows the relative loss of expected interviews at the signal value where minimum reviews occur.''',
    'Decile': '''Post-match applicant and program match distribution for various analyses, depicted through decile plots. '''
}

ANALYSIS_COLORS = sns.color_palette("tab10", len(ANALYSES_TO_GRAPH))

METRIC_STYLES_A = {
    'p_int_given_signal_mean': {'ls': '-', 'marker': 'o'},
    'p_int_given_nosignal_mean': {'ls': '--', 'marker': 'x'},
    'pct_matches_from_signal_mean': {'ls': ':', 'marker': 's'},
    'pct_match_from_nosignal_mean': {'ls': '-.', 'marker': 'd'}
}

ANALYSES_RESIDUALS = [
    "base",
    "random_distribution",
    "random_applicant_rank_list",
    "random_program_rank_list",
    "random_all"
]

@st.cache_data
def load_data(directory):
    data_dict = {}
    for analysis in ANALYSES_TO_GRAPH:
        file_path = os.path.join(directory, f"{analysis}.csv")
        if os.path.exists(file_path):
            data_dict[analysis] = pd.read_csv(file_path)
    return data_dict

@st.cache_data
def load_residual_data(directory):
    all_data = []
    for analysis in ANALYSES_RESIDUALS:
        file_path = os.path.join(directory, f"{analysis}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            processed_df = calculate_residuals(df, analysis)
            all_data.append(processed_df)
    if all_data:
        return pd.concat(all_data, ignore_index=True).sort_values(by='program', ascending=False)
    return pd.DataFrame()

def calculate_residuals(df, analysis_name):
    results = []
    for program, group in df.groupby('program'):
        min_rev_idx = group['reviews_per_program_mean'].idxmin()
        s_min_rev = group.loc[min_rev_idx, 'signals']
        min_reviews = group.loc[min_rev_idx, 'reviews_per_program_mean']

        max_exp_idx = group['expect_int_per_signal_mean'].idxmax()
        s_max_exp = group.loc[max_exp_idx, 'signals']
        max_expected_interviews = group.loc[max_exp_idx, 'expect_int_per_signal_mean']

        rev_at_s_max_exp = group[group['signals'] == s_max_exp]['reviews_per_program_mean'].values[0]
        exp_int_at_s_min_rev = group[group['signals'] == s_min_rev]['expect_int_per_signal_mean'].values[0]

        signal_distance = s_max_exp - s_min_rev

        if min_reviews > 0:
            rel_increase_rev = ((rev_at_s_max_exp - min_reviews) / min_reviews) * 100
        else:
            rel_increase_rev = 0

        if max_expected_interviews > 0:
            loss_at_min_rev = max_expected_interviews - exp_int_at_s_min_rev
            rel_loss_min_rev = (loss_at_min_rev / max_expected_interviews) * 100
        else:
            rel_loss_min_rev = 0

        results.append({
            'program': program,
            'analysis': analysis_name,
            'signal_distance': signal_distance,
            'rel_increase_rev': rel_increase_rev,
            'rel_loss_min_rev': rel_loss_min_rev
        })
    return pd.DataFrame(results)

def create_panel_a(data_dict, program_name):
    fig, ax_a = plt.subplots(figsize=(8, 6))

    analysis_legend_elements = [Line2D([0], [0], color=ANALYSIS_COLORS[i], lw=3, label=ana)
                                for i, ana in enumerate(ANALYSES_TO_GRAPH)]

    for i, analysis in reversed(list(enumerate(ANALYSES_TO_GRAPH))):
        if analysis not in data_dict: continue
        prog_df = data_dict[analysis][data_dict[analysis]['program'] == program_name].sort_values('signals')
        if prog_df.empty: continue
        color = ANALYSIS_COLORS[i]
        for metric, style in METRIC_STYLES_A.items():
            if metric in prog_df.columns:
                ax_a.plot(prog_df['signals'], prog_df[metric] * 100, color=color,
                          linestyle=style['ls'], marker=style['marker'], markersize=5, alpha=0.8)

    ax_a.set_title(PANEL_TITLES[0], fontsize=14)
    ax_a.set_xlabel("Signals")
    ax_a.set_ylabel("Percentage (%)")
    ax_a.grid(True, linestyle='--', alpha=0.6)
    metric_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=style['ls'],
                                     marker=style['marker'], label=METRIC_NAMES[metric])
                              for metric, style in METRIC_STYLES_A.items()]
    leg1 = ax_a.legend(handles=analysis_legend_elements, loc='upper left', title="Analyses", fontsize=9)
    ax_a.add_artist(leg1)
    ax_a.legend(handles=metric_legend_elements, loc='upper right', title="Metrics", fontsize=9)

    plt.tight_layout()
    return fig

def create_panel_b(data_dict, program_name):
    fig, ax_b = plt.subplots(figsize=(8, 6))

    analysis_legend_elements = [Line2D([0], [0], color=ANALYSIS_COLORS[i], lw=3, label=ana)
                                for i, ana in enumerate(ANALYSES_TO_GRAPH)]

    metric_b = 'unfilled_positions_mean'
    for i, analysis in reversed(list(enumerate(ANALYSES_TO_GRAPH))):
        if analysis not in data_dict: continue
        prog_df = data_dict[analysis][data_dict[analysis]['program'] == program_name].sort_values('signals')
        if prog_df.empty: continue
        color = ANALYSIS_COLORS[i]
        if metric_b in prog_df.columns:
            ax_b.plot(prog_df['signals'], prog_df[metric_b], color=color, marker='s', markersize=5)
            if 'unfilled_positions_lower' in prog_df.columns:
                ax_b.fill_between(prog_df['signals'], prog_df['unfilled_positions_lower'],
                                  prog_df['unfilled_positions_upper'], color=color, alpha=0.15)
    ax_b.set_title(PANEL_TITLES[1], fontsize=14)
    ax_b.set_xlabel("Signals")
    ax_b.set_ylabel("Count")
    ax_b.grid(True, linestyle='--', alpha=0.6)
    ax_b.legend(handles=analysis_legend_elements, loc='best')

    plt.tight_layout()
    return fig

def create_panel_c(data_dict, program_name):
    fig, ax_c1 = plt.subplots(figsize=(8, 6))
    ax_c2 = ax_c1.twinx()

    analysis_legend_elements = [Line2D([0], [0], color=ANALYSIS_COLORS[i], lw=3, label=ana)
                                for i, ana in enumerate(ANALYSES_TO_GRAPH)]

    metric_c1 = 'reviews_per_program_mean'
    metric_c2 = 'expect_int_per_signal_mean'
    for i, analysis in reversed(list(enumerate(ANALYSES_TO_GRAPH))):
        if analysis not in data_dict: continue
        prog_df = data_dict[analysis][data_dict[analysis]['program'] == program_name].sort_values('signals')
        if prog_df.empty: continue
        color = ANALYSIS_COLORS[i]
        if metric_c1 in prog_df.columns:
            ax_c1.plot(prog_df['signals'], prog_df[metric_c1], color=color, linestyle='-', marker='^', alpha=0.8)
        if metric_c2 in prog_df.columns:
            ax_c2.plot(prog_df['signals'], prog_df[metric_c2], color=color, linestyle='--', marker='v', alpha=0.8)
    ax_c1.set_ylabel(METRIC_NAMES[metric_c1])
    ax_c2.set_ylabel(METRIC_NAMES[metric_c2])
    ax_c1.set_title(PANEL_TITLES[2], fontsize=14)
    ax_c1.set_xlabel("Signals")
    ax_c1.grid(True, linestyle='--', alpha=0.6)
    c_style_elements = [
        Line2D([0], [0], color='black', linestyle='-', marker='^', label=METRIC_NAMES[metric_c1]),
        Line2D([0], [0], color='black', linestyle='--', marker='v', label=METRIC_NAMES[metric_c2])
    ]
    ax_c1.legend(handles=c_style_elements, loc='upper left', title="Metrics")
    ax_c2.legend(handles=analysis_legend_elements, loc='upper right', title="Analyses")

    plt.tight_layout()
    return fig

def create_single_decile_plot(df, program_name, analysis_name, decile_signal):
    """
    Generates a single decile match heatmap for a specific analysis and signal value.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    prog_df = df[df['program'] == program_name]
    point_data = prog_df[prog_df['signals'] == decile_signal]

    if point_data.empty:
        ax.text(0.5, 0.5, f"No data for {analysis_name}\nat signals = {decile_signal}",
                ha='center', va='center')
        ax.set_title(f"Analysis: {analysis_name}", fontsize=14)
    else:
        decile_matrix = np.zeros((10, 10))
        for p in range(1, 11):
            for a in range(1, 11):
                col_name = f'p{p}_a{a}'
                if col_name in point_data.columns:
                    decile_matrix[p-1, a-1] = point_data.iloc[0][col_name]

        sns.heatmap(decile_matrix, ax=ax, cmap="YlGnBu", annot=False, square=True,
                    vmin=0, cbar_kws={'label': 'Match Probability', 'shrink': 0.8})

        ax.set_title(f"Analysis: {analysis_name}", fontsize=16, fontweight='bold')
        ax.set_xticks(np.arange(10) + 0.5)
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(np.arange(10) + 0.5)
        ax.set_yticklabels(range(1, 11))
        ax.set_xlabel("Applicant Decile")
        ax.set_ylabel("Program Decile")

    plt.tight_layout()
    return fig

def create_residual_graphs(combined_df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 18), sharey=True)
    fig.suptitle("Trade-offs: Minimizing Application Reviews vs. Maximizing Expected Interviews",
                 fontsize=22, fontweight='bold', y=0.96)
    palette = sns.color_palette("tab10", len(ANALYSES_RESIDUALS))

    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    sns.scatterplot(data=combined_df, x='signal_distance', y='program', hue='analysis', palette=palette, s=90, alpha=0.8, ax=ax1, zorder=3)
    ax1.set_title("A. Signal Distance:\nMin Review vs. Max Expected Int.", fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlabel("Difference (Signals)", fontsize=12)
    ax1.set_ylabel("Program", fontsize=14)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, zorder=2)
    ax1.grid(True, axis='x', linestyle=':', alpha=0.6)

    sns.scatterplot(data=combined_df, x='rel_increase_rev', y='program', hue='analysis', palette=palette, s=90, alpha=0.8, ax=ax2, zorder=3)
    ax2.set_title("B. Relative Percent Increase in\nReviews Given Maximized Interviews", fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel("Percent Increase (%)", fontsize=12)
    ax2.set_ylabel("")
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

    sns.scatterplot(data=combined_df, x='rel_loss_min_rev', y='program', hue='analysis', palette=palette, s=90, alpha=0.8, ax=ax3, zorder=3)
    ax3.set_title("C. Relative Percent Loss of Expected\nInterviews Given Minimized Reviews", fontsize=15, fontweight='bold', pad=20)
    ax3.set_xlabel("Percent Loss (%)", fontsize=12)
    ax3.set_ylabel("")
    ax3.grid(True, axis='x', linestyle=':', alpha=0.6)

    unique_programs = combined_df['program'].unique()
    for i, prog in enumerate(unique_programs):
        if i % 2 == 0:
            ax1.axhspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.25, zorder=1)
            ax2.axhspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.25, zorder=1)
            ax3.axhspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.25, zorder=1)

    ax1.set_ylim(-0.5, len(unique_programs) - 0.5)

    if ax1.get_legend(): ax1.get_legend().remove()
    if ax2.get_legend(): ax2.get_legend().remove()
    ax3.legend(title="Sensitivity Analysis", fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.subplots_adjust(top=0.88, wspace=0.05)
    return fig


dir_72 = "results/calculated/gamma_72/"
dir_30 = "results/calculated/gamma_30/"

data_72 = load_data(dir_72)
data_30 = load_data(dir_30)

if "base" in data_72:
    programs = data_72["base"]["program"].unique()
else:
    programs = []

if len(programs) > 0:
    st.header("Panel Graphs")
    selected_program = st.selectbox("Select Program", programs)

    st.subheader(PANEL_TITLES[0])
    st.markdown(f"<p style='color: black; font-size: 1.1rem;'>{Legends['A']}</p>", unsafe_allow_html=True)
    col1_a, col2_a = st.columns(2)
    with col1_a:
        st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 72</p>", unsafe_allow_html=True)
        fig_a_72 = create_panel_a(data_72, selected_program)
        st.pyplot(fig_a_72)
    if data_30 is not None:
        with col2_a:
            st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 30</p>", unsafe_allow_html=True)
            fig_a_30 = create_panel_a(data_30, selected_program)
            st.pyplot(fig_a_30)

    st.subheader(PANEL_TITLES[1])
    st.markdown(f"<p style='color: black; font-size: 1.1rem;'>{Legends['B']}</p>", unsafe_allow_html=True)
    col1_b, col2_b = st.columns(2)
    with col1_b:
        st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 72</p>", unsafe_allow_html=True)
        fig_b_72 = create_panel_b(data_72, selected_program)
        st.pyplot(fig_b_72)
    if data_30 is not None:
        with col2_b:
            st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 30</p>", unsafe_allow_html=True)
            fig_b_30 = create_panel_b(data_30, selected_program)
            st.pyplot(fig_b_30)

    st.subheader(PANEL_TITLES[2])
    st.markdown(f"<p style='color: black; font-size: 1.1rem;'>{Legends['C']}</p>", unsafe_allow_html=True)
    col1_c, col2_c = st.columns(2)
    with col1_c:
        st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 72</p>", unsafe_allow_html=True)
        fig_c_72 = create_panel_c(data_72, selected_program)
        st.pyplot(fig_c_72)
    if data_30 is not None:
        with col2_c:
            st.markdown("<p style='font-size: 1.50rem; font-weight: bold;'>Mean Applications: 30</p>", unsafe_allow_html=True)
            fig_c_30 = create_panel_c(data_30, selected_program)
            st.pyplot(fig_c_30)


st.header("Residual Graphs")
st.markdown(f"<p style='color: black; font-size: 1.1rem;'>{Legends['Residual']}</p>", unsafe_allow_html=True)
col_r1, col_r2 = st.columns(2)

with col_r1:
    st.subheader("Mean Applications: 72")
    res_data_72 = load_residual_data(dir_72)
    if not res_data_72.empty:
        fig_res_72 = create_residual_graphs(res_data_72)
        st.pyplot(fig_res_72)
    else:
        st.write("No residual data available for Mean Applications: 72.")

with col_r2:
    st.subheader("Mean Applications: 30")
    res_data_30 = load_residual_data(dir_30)
    if not res_data_30.empty:
        fig_res_30 = create_residual_graphs(res_data_30)
        st.pyplot(fig_res_30)
    else:
        st.write("No residual data available for Mean Applications: 30.")

if len(programs) > 0:
    st.header("Decile Graphs")
    st.markdown(f"<p style='color: black; font-size: 1.1rem;'>{Legends['Decile']}</p>", unsafe_allow_html=True)

    prog_df_72 = data_72["base"][data_72["base"]["program"] == selected_program]
    signal_options = prog_df_72["signals"].unique()

    col3, col4 = st.columns([1, 3])
    with col3:
        selected_signal = st.selectbox("Select Signal Value", signal_options)
        generate_btn = st.button("Generate")

    if generate_btn:
        st.subheader(f"Side-by-Side Decile Comparisons (72 vs 30) at Signal = {selected_signal}")
        
        for analysis in ANALYSES_TO_GRAPH:
            st.write(f"### Analysis: {analysis}")
            col_dec1, col_dec2 = st.columns(2)
            
            with col_dec1:
                st.write("**Mean Applications: 72**")
                if analysis in data_72:
                    fig_72 = create_single_decile_plot(data_72[analysis], selected_program, analysis, selected_signal)
                    st.pyplot(fig_72)
                else:
                    st.write("Data not found.")
            
            with col_dec2:
                st.write("**Mean Applications: 30**")
                if analysis in data_30:
                    fig_30 = create_single_decile_plot(data_30[analysis], selected_program, analysis, selected_signal)
                    st.pyplot(fig_30)
                else:
                    st.write("Data not found.")
