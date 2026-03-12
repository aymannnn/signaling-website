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
    "random_program_rank_list"
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

def create_panel_abc(data_dict, program_name):
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(f"{program_name} - Cross-Analysis Comparison", fontsize=22, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_c2 = ax_c1.twinx()

    analysis_legend_elements = [Line2D([0], [0], color=ANALYSIS_COLORS[i], lw=3, label=ana)
                                for i, ana in enumerate(ANALYSES_TO_GRAPH)]

    # Panel A
    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
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

    # Panel B
    metric_b = 'unfilled_positions_mean'
    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
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

    # Panel C
    metric_c1 = 'reviews_per_program_mean'
    metric_c2 = 'expect_int_per_signal_mean'
    for i, analysis in enumerate(ANALYSES_TO_GRAPH):
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

def create_panel_d(data_dict, program_name, decile_signal):
    fig, ax_d = plt.subplots(figsize=(8, 6))
    ax_d.set_title(PANEL_TITLES[3].format(decile_signal), fontsize=14)

    if 'base' in data_dict:
        prog_df = data_dict['base'][data_dict['base']['program'] == program_name]
        point_data = prog_df[prog_df['signals'] == decile_signal]

        if point_data.empty:
            ax_d.text(0.5, 0.5, f"No data found for signals = {decile_signal}", ha='center', va='center')
        else:
            decile_matrix = np.zeros((10, 10))
            for p in range(1, 11):
                for a in range(1, 11):
                    col_name = f'p{p}_a{a}'
                    if col_name in point_data.columns:
                        decile_matrix[p-1, a-1] = point_data.iloc[0][col_name]

            sns.heatmap(decile_matrix, ax=ax_d, cmap="YlGnBu", annot=False, vmin=0, cbar_kws={'label': 'Match Probability'})
            ax_d.set_xticks(np.arange(10) + 0.5)
            ax_d.set_xticklabels(range(1, 11))
            ax_d.set_yticks(np.arange(10) + 0.5)
            ax_d.set_yticklabels(range(1, 11))
            ax_d.set_xlabel("Applicant Decile", fontsize=12)
            ax_d.set_ylabel("Program Decile", fontsize=12)
    else:
        ax_d.text(0.5, 0.5, "'base' analysis not found in data", ha='center', va='center')

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

if "show_30" not in st.session_state:
    st.session_state.show_30 = False

if st.button("Display Results for a Mean of 30 Applications"):
    st.session_state.show_30 = not st.session_state.show_30

use_30 = st.session_state.show_30

dir_72 = "results/calculated/gamma_72/"
dir_30 = "results/calculated/gamma_30/"

data_72 = load_data(dir_72)
data_30 = load_data(dir_30) if use_30 else None

if "base" in data_72:
    programs = data_72["base"]["program"].unique()
else:
    programs = []

if len(programs) > 0:
    st.header("Panel Graphs")
    selected_program = st.selectbox("Select Program", programs)

    col1, col2 = st.columns(2) if use_30 else [st.container(), None]

    with col1:
        st.subheader("Gamma 72")
        fig_abc_72 = create_panel_abc(data_72, selected_program)
        st.pyplot(fig_abc_72)

    if use_30 and data_30:
        with col2:
            st.subheader("Gamma 30")
            fig_abc_30 = create_panel_abc(data_30, selected_program)
            st.pyplot(fig_abc_30)

    st.subheader("Decile Graph (Panel D)")

    prog_df_72 = data_72["base"][data_72["base"]["program"] == selected_program]
    signal_options = prog_df_72["signals"].unique()

    col3, col4 = st.columns([1, 3])
    with col3:
        selected_signal = st.selectbox("Select Signal Value", signal_options)
        generate_btn = st.button("Generate")

    if generate_btn:
        col5, col6 = st.columns(2) if use_30 else [st.container(), None]
        with col5:
            st.write("Gamma 72")
            fig_d_72 = create_panel_d(data_72, selected_program, selected_signal)
            st.pyplot(fig_d_72)
        if use_30 and data_30:
            with col6:
                st.write("Gamma 30")
                fig_d_30 = create_panel_d(data_30, selected_program, selected_signal)
                st.pyplot(fig_d_30)

st.header("Residual Graphs")
col_r1, col_r2 = st.columns(2) if use_30 else [st.container(), None]

with col_r1:
    st.subheader("Gamma 72")
    res_data_72 = load_residual_data(dir_72)
    if not res_data_72.empty:
        fig_res_72 = create_residual_graphs(res_data_72)
        st.pyplot(fig_res_72)
    else:
        st.write("No residual data available for Gamma 72.")

if use_30:
    with col_r2:
        st.subheader("Gamma 30")
        res_data_30 = load_residual_data(dir_30)
        if not res_data_30.empty:
            fig_res_30 = create_residual_graphs(res_data_30)
            st.pyplot(fig_res_30)
        else:
            st.write("No residual data available for Gamma 30.")
