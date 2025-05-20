# dashboard.py
# Version: 1.2.0 (Integrating LLM Judge Score and Feedback Display)

import streamlit as st
import sqlite3
import pandas as pd
import json
import os

DB_PATH = "alpha_evolve_programs.db"


# --- MODIFIED: load_program_data (Blueprint Step 6) ---
def load_program_data(task_id_filter=None):  # Method_v1.3.0 (LLM Judge metrics)
    """Loads program data from the SQLite database, optionally filtered by task_id."""
    if not os.path.exists(DB_PATH):
        st.info("Database file 'alpha_evolve_programs.db' not found. Has a run completed yet?")
        return pd.DataFrame()

    df = pd.DataFrame()
    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        # Ensure llm_judge_feedback is selected if it's a column
        # If it's not a column yet because DB wasn't updated, this will fail gracefully or pandas will ignore.
        # For robustness, check if column exists or handle potential error.
        # For now, assume SQLiteDatabaseAgent created it.
        query = "SELECT id, generation, creation_method, parent_id, parent_ids, fitness_scores, errors, code, task_id, llm_judge_feedback FROM programs"  # ADDED llm_judge_feedback
        if task_id_filter:
            query += " WHERE task_id = ?"
            params = (task_id_filter,)
        else:
            params = ()

        # Test if llm_judge_feedback column exists before trying to query it, to be more robust with older DBs
        # For simplicity in this step, we'll assume the column exists due to SQLiteDatabaseAgent updates.
        # A more robust version would query table_info first.

        df = pd.read_sql_query(query, conn, params=params if params else None)

    except sqlite3.OperationalError as e_op:
        if "no such column: llm_judge_feedback" in str(e_op):
            st.warning(
                "Database column 'llm_judge_feedback' not found. Trying to load without it. Please ensure DB schema is updated.")
            try:  # Fallback query without the new column
                query_fallback = "SELECT id, generation, creation_method, parent_id, parent_ids, fitness_scores, errors, code, task_id FROM programs"
                if task_id_filter:
                    query_fallback += " WHERE task_id = ?"
                df = pd.read_sql_query(query_fallback, conn, params=params if params else None)
                df['llm_judge_feedback'] = None  # Add the column as None
            except Exception as e_fallback:
                st.error(f"Error reading from database (fallback query failed): {e_fallback}")
                if conn: conn.close()
                return pd.DataFrame()
        else:
            st.error(f"Error reading from database: {e_op}")
            if conn: conn.close()
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading from database (table 'programs' might be missing or schema is incorrect): {e}")
        if conn: conn.close()
        return pd.DataFrame()
    finally:
        if conn: conn.close()

    if df.empty:  # Handle case where df might be empty after query
        # Initialize columns to prevent downstream errors even if df is empty
        df['correctness'] = pd.Series(dtype='float64')
        df['runtime_ms'] = pd.Series(dtype='float64')
        df['ruff_violations'] = pd.Series(dtype='float64')
        df['llm_judge_overall_score'] = pd.Series(dtype='float64')  # NEW
        df['llm_judge_feedback'] = pd.Series(dtype='object')  # Ensure column exists
        df['parent_ids_display'] = pd.Series(dtype='object')
        return df

    if 'fitness_scores' in df.columns:
        parsed_scores_list = []
        for score_str in df['fitness_scores']:
            try:
                if score_str:
                    parsed_scores_list.append(json.loads(score_str))
                else:
                    parsed_scores_list.append({})
            except (json.JSONDecodeError, TypeError):
                parsed_scores_list.append({})

        score_df = pd.DataFrame(parsed_scores_list).reindex(df.index, fill_value={})

        df['correctness'] = score_df.get('correctness', pd.Series(0.0, index=df.index)) * 100
        df['runtime_ms'] = score_df.get('runtime_ms', pd.Series(float('inf'), index=df.index))
        df['ruff_violations'] = score_df.get('ruff_violations', pd.Series(float('inf'), index=df.index))
        # --- NEW: Parse LLM Judge Score ---
        df['llm_judge_overall_score'] = score_df.get('llm_judge_overall_score',
                                                     pd.Series(0.0, index=df.index))  # Default to 0 if not found

    elif not df.empty:  # fitness_scores column missing
        st.warning("Column 'fitness_scores' not found. Cannot parse detailed scores including LLM Judge score.")
        df['correctness'] = 0.0
        df['runtime_ms'] = float('inf')
        df['ruff_violations'] = float('inf')
        df['llm_judge_overall_score'] = 0.0  # NEW

    # Ensure llm_judge_feedback column exists if not loaded by query (e.g. due to fallback)
    if 'llm_judge_feedback' not in df.columns and not df.empty:
        df['llm_judge_feedback'] = None  # Initialize if missing

    if 'parent_ids' in df.columns and not df.empty:
        df['parent_ids_display'] = df['parent_ids'].apply(
            lambda x: ', '.join(json.loads(x)) if x and x not in ['null', 'None', None] else None
        )
    elif 'parent_ids_display' not in df.columns and not df.empty:
        df['parent_ids_display'] = pd.Series(dtype='object')

    return df


def load_run_metrics_data(task_id_filter=None):
    metrics_file = "run_metrics.jsonl"
    if not os.path.exists(metrics_file):
        st.info(f"Metrics file '{metrics_file}' not found. Run an experiment to generate it.")
        return pd.DataFrame()
    records = []
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    st.warning(f"Skipping invalid JSON line in {metrics_file}: {line.strip()}")
    except Exception as e:
        st.error(f"Error reading {metrics_file}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if task_id_filter and not df.empty and 'task_id' in df.columns:
        df = df[df['task_id'] == task_id_filter]
    return df


st.set_page_config(page_title="OpenAlpha_Evolve Dashboard", layout="wide")
st.title("🚀 OpenAlpha_Evolve Results Dashboard")

task_ids = ["All Tasks"]
if os.path.exists(DB_PATH):
    conn_tasks = None
    try:
        conn_tasks = sqlite3.connect(DB_PATH)
        task_df = pd.read_sql_query("SELECT DISTINCT task_id FROM programs WHERE task_id IS NOT NULL", conn_tasks)
        if not task_df.empty:
            task_ids.extend(task_df['task_id'].unique().tolist())
    except Exception:
        pass
    finally:
        if conn_tasks: conn_tasks.close()

selected_task_id = st.sidebar.selectbox("Select Task ID to View:", options=task_ids)
task_id_for_filter = selected_task_id if selected_task_id and selected_task_id != "All Tasks" else None

programs_df = load_program_data(task_id_for_filter)
run_metrics_df = load_run_metrics_data(task_id_for_filter)

st.header("📊 Run Summary")
if not programs_df.empty:
    num_programs = len(programs_df)
    best_correctness_overall = programs_df['correctness'].max() if 'correctness' in programs_df and not programs_df[
        'correctness'].empty else 0
    min_ruff_violations_overall = programs_df['ruff_violations'].min() if 'ruff_violations' in programs_df and not \
    programs_df['ruff_violations'].empty else float('inf')
    # --- NEW: LLM Judge Score Summary ---
    best_llm_judge_score_overall = programs_df[
        'llm_judge_overall_score'].max() if 'llm_judge_overall_score' in programs_df and not programs_df[
        'llm_judge_overall_score'].empty else 0.0

    col1, col2, col3 = st.columns(3)  # Added a third column
    with col1:
        st.metric("Total Programs in DB (for task)", num_programs)
    with col2:
        st.metric("Best Correctness Achieved (for task)", f"{best_correctness_overall:.2f}%")
    with col3:  # NEW Column for LLM Judge Score
        st.metric("Best LLM Judge Score (for task)", f"{best_llm_judge_score_overall:.2f}/10")

    if 'ruff_violations' in programs_df.columns:
        st.metric("Minimum Ruff Violations (for task)",
                  f"{min_ruff_violations_overall if min_ruff_violations_overall != float('inf') else 'N/A'}")
else:
    st.info("No program data found. Run an experiment first!")

# --- MODIFIED: Generational Progress Plots (Blueprint Step 6) ---
if not run_metrics_df.empty:
    st.header("📈 Generational Progress")

    metrics_to_plot = {
        'avg_correctness_perc': 'Average Correctness (%)',
        'best_correctness_perc': 'Best Correctness (%)',
        'avg_ruff_violations': 'Average Ruff Violations (Lower is Better)',
        'min_ruff_violations': 'Minimum Ruff Violations (Lower is Better)',
        # --- NEW: LLM Judge Score Plots ---
        'avg_llm_judge_score': 'Average LLM Judge Score (/10)',  # Assuming MonitoringAgent logs this
        'best_llm_judge_score': 'Best LLM Judge Score (/10)',  # Assuming MonitoringAgent logs this
        # --- END NEW ---
        'llm_api_calls_generation': 'LLM API Calls per Gen'
    }

    if 'avg_correctness' in run_metrics_df.columns:
        run_metrics_df['avg_correctness_perc'] = run_metrics_df['avg_correctness'] * 100
    if 'best_correctness' in run_metrics_df.columns:
        run_metrics_df['best_correctness_perc'] = run_metrics_df['best_correctness'] * 100

    # Filter metrics_to_plot to only those present in run_metrics_df columns
    plottable_metrics = {k: v for k, v in metrics_to_plot.items() if k in run_metrics_df.columns}

    if plottable_metrics:
        num_metric_cols = len(plottable_metrics)
        cols = st.columns(num_metric_cols if num_metric_cols > 0 else 1)  # Ensure at least 1 col
        col_idx = 0
        for metric_key, chart_title in plottable_metrics.items():
            with cols[col_idx % num_metric_cols]:  # Use modulo if fewer columns than metrics
                st.subheader(chart_title)
                if 'generation_number' in run_metrics_df.columns:
                    st.line_chart(run_metrics_df.set_index('generation_number')[metric_key])
                else:  # Fallback if generation_number is missing (should not happen)
                    st.line_chart(run_metrics_df[metric_key])
            col_idx += 1
    else:
        st.info(
            "No plottable metrics found in run_metrics.jsonl. Ensure TaskManagerAgent and MonitoringAgent are logging them.")

# --- MODIFIED: Best Programs Table & Detail View (Blueprint Step 6) ---
st.header("🏆 Top Evolved Programs")
if not programs_df.empty:
    default_display_cols = ['id', 'generation', 'creation_method', 'parent_ids_display',
                            'llm_judge_overall_score',  # NEW: Add to default display
                            'correctness', 'ruff_violations', 'runtime_ms', 'task_id']
    available_cols_in_df = [col for col in default_display_cols if col in programs_df.columns]

    selected_display_columns = st.sidebar.multiselect(
        "Select columns for Top Programs table:",
        options=list(programs_df.columns),
        default=available_cols_in_df
    )
    if not selected_display_columns:
        selected_display_columns = available_cols_in_df if available_cols_in_df else ['id']

    # Add llm_judge_overall_score to sort options
    sort_by_options = [col for col in
                       ['llm_judge_overall_score', 'correctness', 'ruff_violations', 'runtime_ms', 'generation'] if
                       col in programs_df.columns]
    default_sort = []
    if 'llm_judge_overall_score' in sort_by_options: default_sort.append(
        'llm_judge_overall_score')  # Default sort by judge score
    if 'correctness' in sort_by_options and 'llm_judge_overall_score' not in default_sort: default_sort.append(
        'correctness')

    sort_by = st.sidebar.multiselect(
        "Sort top programs by:",
        options=sort_by_options,
        default=default_sort
    )

    ascending_map_list = []
    for col_name in sort_by:
        default_asc = True
        if col_name == 'correctness' or col_name == 'llm_judge_overall_score':  # Higher is better for these
            default_asc = False
        # For ruff_violations and runtime_ms, lower is better, so True (ascending) is the "better" sort direction

        # Let user choose ascending for each selected sort column
        if st.sidebar.checkbox(f"Sort '{col_name}' Ascending?", value=default_asc, key=f"sort_asc_{col_name}"):
            ascending_map_list.append(True)
        else:
            ascending_map_list.append(False)

    if sort_by and len(sort_by) == len(ascending_map_list):
        sorted_df = programs_df.sort_values(by=sort_by, ascending=ascending_map_list)
    else:
        # Fallback sort if no user selection or mismatch
        fallback_sort_cols = []
        fallback_asc = []
        if 'llm_judge_overall_score' in programs_df.columns:
            fallback_sort_cols.append('llm_judge_overall_score'); fallback_asc.append(False)
        elif 'correctness' in programs_df.columns:
            fallback_sort_cols.append('correctness'); fallback_asc.append(False)
        if 'ruff_violations' in programs_df.columns: fallback_sort_cols.append('ruff_violations'); fallback_asc.append(
            True)

        if fallback_sort_cols:
            sorted_df = programs_df.sort_values(by=fallback_sort_cols, ascending=fallback_asc)
        else:
            sorted_df = programs_df  # No sorting if no relevant columns

    st.dataframe(sorted_df[selected_display_columns].head(20))

    st.subheader("🔍 View Program Detail")
    program_ids_for_selectbox = [""] + (programs_df['id'].tolist() if 'id' in programs_df.columns else [])
    selected_program_id = st.selectbox("Select Program ID to View Details:", options=program_ids_for_selectbox)

    if selected_program_id and 'id' in programs_df.columns:
        program_detail_series = programs_df[programs_df['id'] == selected_program_id]
        if not program_detail_series.empty:
            program_detail = program_detail_series.iloc[0]
            st.text_area("Code:", value=program_detail.get('code', 'N/A'), height=300)

            # --- NEW: Display LLM Judge Score and Feedback ---
            judge_score = program_detail.get('llm_judge_overall_score', 'N/A')
            judge_feedback_text = program_detail.get('llm_judge_feedback', 'No LLM judge feedback recorded.')

            st.metric("LLM Judge Overall Score", f"{judge_score}/10" if isinstance(judge_score, (
            int, float)) and judge_score != 0.0 else "N/A")  # Check if 0.0 was default for not found

            st.subheader("LLM Judge Feedback:")
            if judge_feedback_text and judge_feedback_text != 'No LLM judge feedback recorded.':
                st.text_area("Feedback:", value=judge_feedback_text, height=150, disabled=True)
            else:
                st.info("No LLM judge feedback recorded for this program.")
            # --- END NEW ---

            st.subheader("Fitness Scores & Other Details:")
            fitness_scores_str = program_detail.get('fitness_scores', '{}')
            try:
                fitness_scores_dict = json.loads(fitness_scores_str) if isinstance(fitness_scores_str,
                                                                                   str) else fitness_scores_str
                st.json(fitness_scores_dict)  # This will show llm_judge_overall_score within the JSON too
            except:
                st.text(f"Raw fitness_scores: {fitness_scores_str}")

            errors_str = program_detail.get('errors', '[]')
            if errors_str and errors_str != '[]':
                st.subheader("Errors Reported:")
                try:
                    errors_list = json.loads(errors_str) if isinstance(errors_str, str) else errors_str
                    if isinstance(errors_list, list) and errors_list:
                        for err_item in errors_list: st.error(str(err_item))
                    elif errors_list:
                        st.text(str(errors_list))
                    else:
                        st.success("No errors reported for this program during evaluation (errors list was empty).")
                except:
                    st.text(f"Raw errors: {errors_str}")
            else:
                st.success("No errors reported for this program during evaluation.")
        else:
            st.warning(f"Program ID {selected_program_id} not found in the current filtered data.")
else:
    st.info("No programs to display in the table for the selected task.")