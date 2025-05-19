import streamlit as st
import sqlite3
import pandas as pd
import json
import os

DB_PATH = "alpha_evolve_programs.db"  # Should be fine if EvaluatorAgent saves to this DB name via SQLiteDatabaseAgent settings


# --- MODIFIED: load_program_data (v1.2.0 for Ruff) ---
def load_program_data(task_id_filter=None):  # Method_v1.2.0 (Ruff metrics)
    """Loads program data from the SQLite database, optionally filtered by task_id."""
    if not os.path.exists(DB_PATH):
        st.info("Database file 'alpha_evolve_programs.db' not found. Has a run completed yet?")
        return pd.DataFrame()

    df = pd.DataFrame()
    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT id, generation, creation_method, parent_id, parent_ids, fitness_scores, errors, code, task_id FROM programs"
        if task_id_filter:
            query += " WHERE task_id = ?"
            params = (task_id_filter,)
        else:
            params = ()
        df = pd.read_sql_query(query, conn, params=params if params else None)
    except Exception as e:
        st.error(f"Error reading from database (table 'programs' might be missing or schema is incorrect): {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        return pd.DataFrame()
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    if not df.empty and 'fitness_scores' in df.columns:
        parsed_scores_list = []  # Changed name for clarity
        for score_str in df['fitness_scores']:
            try:
                if score_str:  # Ensure score_str is not None or empty
                    parsed_scores_list.append(json.loads(score_str))
                else:
                    parsed_scores_list.append({})  # Append empty dict if score_str is None/empty
            except (json.JSONDecodeError, TypeError):
                parsed_scores_list.append({})  # Default for parsing errors

        score_df = pd.DataFrame(parsed_scores_list)
        # Ensure index alignment if parsed_scores_list was shorter due to errors
        score_df = score_df.reindex(df.index, fill_value={})

        df['correctness'] = score_df.get('correctness', pd.Series(0.0, index=df.index)) * 100
        df['runtime_ms'] = score_df.get('runtime_ms', pd.Series(float('inf'), index=df.index))
        # --- Ruff metric instead of Pylint ---
        df['ruff_violations'] = score_df.get('ruff_violations',
                                             pd.Series(float('inf'), index=df.index))  # Lower is better
        # df['pylint_score'] = score_df.get('pylint_score', pd.Series(-10.0, index=df.index)) # Removed
    elif not df.empty:
        st.warning("Column 'fitness_scores' not found. Cannot parse detailed scores.")
        # Initialize columns to prevent downstream errors
        df['correctness'] = 0.0
        df['runtime_ms'] = float('inf')
        df['ruff_violations'] = float('inf')
    else:  # df is empty
        df['correctness'] = pd.Series(dtype='float64')
        df['runtime_ms'] = pd.Series(dtype='float64')
        df['ruff_violations'] = pd.Series(dtype='float64')
        df['parent_ids_display'] = pd.Series(dtype='object')

    if not df.empty and 'parent_ids' in df.columns:
        df['parent_ids_display'] = df['parent_ids'].apply(
            lambda x: ', '.join(json.loads(x)) if x and x != 'null' else None
        )
    elif not df.empty and 'parent_ids_display' not in df.columns:
        df['parent_ids_display'] = pd.Series(dtype='object')

    return df


# load_run_metrics_data: This function loads from run_metrics.jsonl
# It will pick up new fields like "avg_ruff_violations" if TaskManagerAgent puts them there.
# No direct code change needed here if the keys in the JSONL match the new metric names.
def load_run_metrics_data(task_id_filter=None):
    """Loads generational metrics from run_metrics.jsonl"""
    metrics_file = "run_metrics.jsonl"  # This should match MonitoringAgent's output
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


# --- Streamlit App Layout ---
st.set_page_config(page_title="OpenAlpha_Evolve Dashboard", layout="wide")
st.title("🚀 OpenAlpha_Evolve Results Dashboard")

task_ids = ["All Tasks"]
if os.path.exists(DB_PATH):
    conn_tasks = None
    try:
        conn_tasks = sqlite3.connect(DB_PATH)
        task_df = pd.read_sql_query("SELECT DISTINCT task_id FROM programs WHERE task_id IS NOT NULL",
                                    conn_tasks)  # Added WHERE task_id IS NOT NULL
        if not task_df.empty:
            task_ids.extend(task_df['task_id'].unique().tolist())  # Use unique() for safety
    except Exception:
        pass  # Table might not exist, or other read error
    finally:
        if conn_tasks: conn_tasks.close()

selected_task_id = st.sidebar.selectbox("Select Task ID to View:", options=task_ids)
task_id_for_filter = selected_task_id if selected_task_id and selected_task_id != "All Tasks" else None

programs_df = load_program_data(task_id_for_filter)
run_metrics_df = load_run_metrics_data(task_id_for_filter)  # This will load new ruff metrics if present in JSONL

st.header("📊 Run Summary")
if not programs_df.empty:
    num_programs = len(programs_df)
    best_correctness_overall = programs_df['correctness'].max() if 'correctness' in programs_df and not programs_df[
        'correctness'].empty else 0
    # Add a summary for Ruff violations if available
    min_ruff_violations_overall = programs_df['ruff_violations'].min() if 'ruff_violations' in programs_df and not \
    programs_df['ruff_violations'].empty else float('inf')

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Programs in DB (for task)", num_programs)
    with col2:
        st.metric("Best Correctness Achieved (for task)", f"{best_correctness_overall:.2f}%")
    if 'ruff_violations' in programs_df.columns:  # Only show if column exists
        st.metric("Minimum Ruff Violations (for task)",
                  f"{min_ruff_violations_overall if min_ruff_violations_overall != float('inf') else 'N/A'}")
else:
    st.info("No program data found. Run an experiment first!")

# --- MODIFIED: Generational Progress Plots (v1.1.0 for Ruff) ---
if not run_metrics_df.empty:
    st.header("📈 Generational Progress")

    metrics_to_plot = {
        'avg_correctness_perc': 'Average Correctness (%)',  # Will be created if avg_correctness exists
        'best_correctness_perc': 'Best Correctness (%)',  # Will be created if best_correctness exists
        # --- Ruff metrics instead of Pylint ---
        'avg_ruff_violations': 'Average Ruff Violations (Lower is Better)',
        'min_ruff_violations': 'Minimum Ruff Violations (Lower is Better)',
        # 'avg_pylint_score': 'Average Pylint Score', # Removed
        'llm_api_calls_generation': 'LLM API Calls per Gen'
    }

    # Prepare percentage columns for correctness if original 0-1 scale metrics exist
    if 'avg_correctness' in run_metrics_df.columns:
        run_metrics_df['avg_correctness_perc'] = run_metrics_df['avg_correctness'] * 100
    if 'best_correctness' in run_metrics_df.columns:
        run_metrics_df['best_correctness_perc'] = run_metrics_df['best_correctness'] * 100

    num_metric_cols = len([m for m in metrics_to_plot if
                           m in run_metrics_df.columns or (m.endswith('_perc') and m[:-5] in run_metrics_df.columns)])
    if num_metric_cols > 0:
        cols = st.columns(num_metric_cols)
        col_idx = 0
        for metric_key, chart_title in metrics_to_plot.items():
            if metric_key in run_metrics_df.columns:
                with cols[col_idx]:
                    st.subheader(chart_title)
                    if 'generation_number' in run_metrics_df.columns:
                        st.line_chart(run_metrics_df.set_index('generation_number')[metric_key])
                    else:
                        st.line_chart(run_metrics_df[metric_key])
                col_idx += 1
            # Handle cases where _perc columns were defined but original might be missing (though unlikely with above prep)
            elif metric_key.endswith('_perc') and metric_key[:-5] in run_metrics_df.columns:
                pass  # Already handled by plotting the _perc column if it exists
            elif not (metric_key.endswith('_perc')):  # Don't warn for _perc if original was processed
                st.sidebar.warning(f"Metric '{metric_key}' not found in run_metrics.jsonl for plotting.")
    else:
        st.info("No plottable metrics found in run_metrics.jsonl. Ensure TaskManagerAgent is logging them.")

# --- MODIFIED: Best Programs Table (v1.1.0 for Ruff) ---
st.header("🏆 Top Evolved Programs")
if not programs_df.empty:
    # Default display columns, updated for Ruff
    default_display_cols = ['id', 'generation', 'creation_method', 'parent_ids_display', 'correctness',
                            'ruff_violations', 'runtime_ms', 'task_id']
    # Filter to actual available columns to prevent errors if DB schema is old or some metrics weren't calculated
    available_cols_in_df = [col for col in default_display_cols if col in programs_df.columns]

    # User can select columns to display
    selected_display_columns = st.sidebar.multiselect(
        "Select columns for Top Programs table:",
        options=list(programs_df.columns),  # Offer all available columns from the DataFrame
        default=available_cols_in_df  # Default to our curated list if columns exist
    )
    if not selected_display_columns:  # Ensure some columns are selected
        selected_display_columns = available_cols_in_df if available_cols_in_df else ['id']

    sort_by_options = [col for col in ['correctness', 'ruff_violations', 'runtime_ms', 'generation'] if
                       col in programs_df.columns]
    default_sort = []
    if 'correctness' in sort_by_options: default_sort.append('correctness')
    if 'ruff_violations' in sort_by_options: default_sort.append('ruff_violations')

    sort_by = st.sidebar.multiselect(
        "Sort top programs by:",
        options=sort_by_options,
        default=default_sort
    )

    ascending_map_list = []
    for col_name in sort_by:
        # Ruff violations: lower is better (so Ascending = True is preferred)
        # Correctness: higher is better (so Ascending = False is preferred)
        # Runtime: lower is better (Ascending = True is preferred)
        default_asc = True  # Default to ascending
        if col_name == 'correctness':
            default_asc = False

        if st.sidebar.checkbox(f"Sort '{col_name}' Ascending?", value=default_asc, key=f"sort_asc_{col_name}"):
            ascending_map_list.append(True)
        else:
            ascending_map_list.append(False)

    if sort_by and len(sort_by) == len(ascending_map_list):  # Ensure lists match
        sorted_df = programs_df.sort_values(by=sort_by, ascending=ascending_map_list)
    else:  # Default sort if no user selection or mismatch
        # Fallback to sorting by correctness (desc) and ruff_violations (asc) if available
        fallback_sort_cols = []
        fallback_asc = []
        if 'correctness' in programs_df.columns: fallback_sort_cols.append('correctness'); fallback_asc.append(False)
        if 'ruff_violations' in programs_df.columns: fallback_sort_cols.append('ruff_violations'); fallback_asc.append(
            True)
        if fallback_sort_cols:
            sorted_df = programs_df.sort_values(by=fallback_sort_cols, ascending=fallback_asc)
        else:
            sorted_df = programs_df

    st.dataframe(sorted_df[selected_display_columns].head(20))

    st.subheader("🔍 View Program Detail")
    program_ids_for_selectbox = [""] + (programs_df['id'].tolist() if 'id' in programs_df.columns else [])
    selected_program_id = st.selectbox("Select Program ID to View Details:", options=program_ids_for_selectbox)

    if selected_program_id and 'id' in programs_df.columns:  # Ensure 'id' column exists
        program_detail_series = programs_df[programs_df['id'] == selected_program_id]
        if not program_detail_series.empty:
            program_detail = program_detail_series.iloc[0]
            st.text_area("Code:", value=program_detail.get('code', 'N/A'), height=400)

            # Display fitness_scores JSON
            fitness_scores_str = program_detail.get('fitness_scores', '{}')
            try:
                fitness_scores_dict = json.loads(fitness_scores_str) if isinstance(fitness_scores_str,
                                                                                   str) else fitness_scores_str
                st.json(fitness_scores_dict)
            except:  # Fallback if it's not a valid JSON string for some reason
                st.text(f"Raw fitness_scores: {fitness_scores_str}")

            errors_str = program_detail.get('errors', '[]')
            if errors_str and errors_str != '[]':  # Check if errors string is not empty or just '[]'
                st.subheader("Errors Reported:")
                try:
                    errors_list = json.loads(errors_str) if isinstance(errors_str, str) else errors_str
                    if isinstance(errors_list, list) and errors_list:
                        for err_item in errors_list:
                            st.error(str(err_item))  # Display each error
                    elif errors_list:  # Not a list but not empty
                        st.text(str(errors_list))
                    else:  # Parsed to empty list or was empty
                        st.success("No errors reported for this program during evaluation (errors list was empty).")
                except:  # Fallback if errors_str is not parsable JSON list
                    st.text(f"Raw errors: {errors_str}")
            else:
                st.success("No errors reported for this program during evaluation.")
        else:
            st.warning(f"Program ID {selected_program_id} not found in the current filtered data.")
else:
    st.info("No programs to display in the table for the selected task.")