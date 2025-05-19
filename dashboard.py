import streamlit as st
import sqlite3
import pandas as pd  # Keep this
import json
import os

DB_PATH = "alpha_evolve_programs.db"


def load_program_data(task_id_filter=None):
    """Loads program data from the SQLite database, optionally filtered by task_id."""
    if not os.path.exists(DB_PATH):
        st.info("Database file 'alpha_evolve_programs.db' not found. Has a run completed yet?")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, generation, creation_method, parent_id, parent_ids, fitness_scores, errors, code, task_id FROM programs"  # Added parent_id, parent_ids
    if task_id_filter:
        # Use parameterized query to prevent SQL injection, though less critical for local trusted input
        query += " WHERE task_id = ?"
        params = (task_id_filter,)
    else:
        params = ()

    try:
        # Pass params to read_sql_query if they exist
        df = pd.read_sql_query(query, conn, params=params if params else None)
    except Exception as e:  # Catch a more general Exception from pandas or sqlite3
        st.error(f"Error reading from database (table 'programs' might be missing or schema is incorrect): {e}")
        st.info(
            "Please ensure an evolutionary run has completed successfully to create/populate the database with the correct schema.")
        # We should still close the connection if it was opened
        conn.close()
        return pd.DataFrame()  # Return empty dataframe on error
    finally:
        # Ensure connection is closed if it was opened and no error occurred before this.
        # If an error occurred and conn.close() was already called, this is fine.
        # A more robust way is to use `with sqlite3.connect(DB_PATH) as conn:` if not using pandas directly for connection.
        # However, pandas usually handles connection management for read_sql_query if given a path,
        # but here we're managing it manually.
        if 'conn' in locals() and conn:  # Check if conn was defined and potentially still open
            try:
                # Check if connection is still active before trying to close
                # This is a bit of a heuristic as sqlite3 connections don't have a direct 'is_closed'
                conn.execute("SELECT 1")  # Try a simple command
                conn.close()
            except sqlite3.ProgrammingError:  # Connection already closed or unusable
                pass
            except Exception:  # Other potential errors on close
                pass

    # Parse JSON strings in fitness_scores and add key metrics as columns
    if not df.empty and 'fitness_scores' in df.columns:
        parsed_scores = []
        for score_str in df['fitness_scores']:
            try:
                parsed_scores.append(json.loads(score_str))
            except (json.JSONDecodeError, TypeError):
                parsed_scores.append({})

        score_df = pd.DataFrame(parsed_scores)
        df['correctness'] = score_df.get('correctness', pd.Series(0.0, index=df.index)) * 100
        df['runtime_ms'] = score_df.get('runtime_ms', pd.Series(float('inf'), index=df.index))
        df['pylint_score'] = score_df.get('pylint_score', pd.Series(-10.0, index=df.index))
        # Add more parsed metrics as needed
    elif not df.empty:  # df exists but no fitness_scores column (schema issue?)
        st.warning("Column 'fitness_scores' not found in the loaded program data. Cannot parse detailed scores.")
        # Initialize to prevent errors later if these columns are expected
        df['correctness'] = pd.Series(dtype='float64')
        df['runtime_ms'] = pd.Series(dtype='float64')
        df['pylint_score'] = pd.Series(dtype='float64')
    else:  # df is empty, initialize columns
        df['correctness'] = pd.Series(dtype='float64')
        df['runtime_ms'] = pd.Series(dtype='float64')
        df['pylint_score'] = pd.Series(dtype='float64')

    # Handle parent_ids parsing if you want to display it cleanly
    if not df.empty and 'parent_ids' in df.columns:
        # This column contains JSON strings of lists, e.g., "[\"parent1_id\", \"parent2_id\"]" or "null"
        # You might want to display it as a comma-separated string or similar.
        df['parent_ids_display'] = df['parent_ids'].apply(
            lambda x: ', '.join(json.loads(x)) if x and x != 'null' else None
        )
    elif not df.empty:
        df['parent_ids_display'] = pd.Series(dtype='object')

    return df


def load_run_metrics_data(task_id_filter=None):
    """Loads generational metrics from run_metrics.jsonl"""
    metrics_file = "run_metrics.jsonl"
    if not os.path.exists(metrics_file):
        return pd.DataFrame()

    records = []
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
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

# --- Task Selector ---
# Get unique task_ids from the database for the selector
task_ids = ["All Tasks"]
if os.path.exists(DB_PATH):
    conn_tasks = sqlite3.connect(DB_PATH)
    try:
        task_df = pd.read_sql_query("SELECT DISTINCT task_id FROM programs", conn_tasks)
        if not task_df.empty:
            task_ids.extend(task_df['task_id'].tolist())
    except Exception:  # Table might not exist yet
        pass
    finally:
        conn_tasks.close()

selected_task_id = st.sidebar.selectbox("Select Task ID to View:", options=task_ids)

task_id_for_filter = None
if selected_task_id and selected_task_id != "All Tasks":
    task_id_for_filter = selected_task_id

# --- Load Data based on selection ---
programs_df = load_program_data(task_id_for_filter)
run_metrics_df = load_run_metrics_data(task_id_for_filter)

# --- Display Summary ---
st.header("📊 Run Summary")
if not programs_df.empty:
    num_programs = len(programs_df)
    best_correctness_overall = programs_df['correctness'].max() if 'correctness' in programs_df and not programs_df[
        'correctness'].empty else 0
    st.metric("Total Programs in DB (for task)", num_programs)
    st.metric("Best Correctness Achieved (for task)", f"{best_correctness_overall:.2f}%")
else:
    st.info(
        "No program data found for the selected task (or database is empty/table missing). Run an experiment first!")

# --- Generational Progress Plots ---
if not run_metrics_df.empty:
    st.header("📈 Generational Progress")

    # Ensure columns exist before trying to plot
    metrics_to_plot = {
        'avg_correctness': 'Average Correctness (%)',
        'best_correctness': 'Best Correctness (%)',
        'avg_pylint_score': 'Average Pylint Score',
        'llm_api_calls_generation': 'LLM API Calls per Gen'
    }
    # Convert correctness to percentage for plotting if it's not already
    if 'avg_correctness' in run_metrics_df.columns:
        run_metrics_df['avg_correctness_perc'] = run_metrics_df['avg_correctness'] * 100
        metrics_to_plot['avg_correctness_perc'] = 'Average Correctness (%)'
        del metrics_to_plot['avg_correctness']  # remove the original after creating perc
    if 'best_correctness' in run_metrics_df.columns:
        run_metrics_df['best_correctness_perc'] = run_metrics_df['best_correctness'] * 100
        metrics_to_plot['best_correctness_perc'] = 'Best Correctness (%)'
        del metrics_to_plot['best_correctness']

    cols = st.columns(len(metrics_to_plot))
    col_idx = 0
    for metric_key, chart_title in metrics_to_plot.items():
        if metric_key in run_metrics_df.columns:
            with cols[col_idx]:
                st.subheader(chart_title)
                # Use 'generation_number' for x-axis if it exists
                if 'generation_number' in run_metrics_df.columns:
                    st.line_chart(run_metrics_df.set_index('generation_number')[metric_key])
                else:
                    st.line_chart(run_metrics_df[metric_key])  # Plot against index if no gen number
            col_idx += 1
        else:
            st.warning(f"Metric '{metric_key}' not found in run_metrics.jsonl for plotting.")

# --- Best Programs Table ---
st.header("🏆 Top Evolved Programs")
if not programs_df.empty:
    # Select columns to display and sort
    display_columns = ['id', 'generation', 'creation_method', 'parent id', 'correctness', 'runtime_ms', 'pylint_score', 'task_id']
    # Filter out columns that might not exist yet if df is very new
    actual_display_columns = [col for col in display_columns if col in programs_df.columns]

    sort_by = st.sidebar.multiselect(
        "Sort top programs by:",
        options=actual_display_columns,  # Only allow sorting by available columns
        default=['correctness', 'runtime_ms']
    )

    ascending_map_list = []
    for col_name in sort_by:
        # Default sort order (True for ascending, False for descending)
        default_asc = True
        if col_name in ['correctness', 'pylint_score']:  # Higher is better
            default_asc = False

        # Let user choose sort order for each selected column
        if st.sidebar.checkbox(f"Sort '{col_name}' Ascending?", value=default_asc, key=f"sort_asc_{col_name}"):
            ascending_map_list.append(True)
        else:
            ascending_map_list.append(False)

    if sort_by:
        sorted_df = programs_df.sort_values(by=sort_by, ascending=ascending_map_list)
    else:
        sorted_df = programs_df  # Default sort if nothing selected (usually by index)

    st.dataframe(sorted_df[actual_display_columns].head(20))  # Show top 20

    # --- Program Detail Viewer ---
    st.subheader("🔍 View Program Detail")
    program_ids = [""] + programs_df['id'].tolist()  # Add an empty option
    selected_program_id = st.selectbox("Select Program ID to View Details:", options=program_ids)

    if selected_program_id:
        program_detail = programs_df[programs_df['id'] == selected_program_id].iloc[0]
        st.text_area("Code:", value=program_detail['code'], height=400)
        st.json(program_detail['fitness_scores'])  # Show raw fitness JSON
        if program_detail['errors'] and program_detail['errors'] != '[]':
            st.subheader("Errors:")
            try:
                errors_list = json.loads(program_detail['errors'])
                for err in errors_list:
                    st.error(err)
            except:
                st.text(program_detail['errors'])  # Show raw string if not parsable
        else:
            st.success("No errors reported for this program during evaluation.")
else:
    st.info("No programs to display in the table.")