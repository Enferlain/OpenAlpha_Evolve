        import json
        import time
        import sys
        import math

        # User's code (function to be tested)
        import heapq

def dijkstra(graph, source_node):
    distances = {node: float('inf') for node in graph}
    distances[source_node] = 0
    priority_queue = [(0, source_node)]

    while priority_queue:
        distance, current_node = heapq.heappop(priority_queue)

        if distance > distances[current_node]:
            continue

        for neighbor, weight in graph.get(current_node, {}).items():
            new_distance = distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return distances

        # --- Helper function to convert string keys back to int if possible ---
        def int_keys_if_possible(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    try:
                        # Try to convert key to int. If it's not an int-like string, keep original.
                        # This is a simple heuristic for graph node IDs.
                        # More robust: check if all keys in a specific 'graph' dict are int-like.
                        int_k = int(k)
                        new_dict[int_k] = int_keys_if_possible(v)
                    except ValueError:
                        new_dict[k] = int_keys_if_possible(v) # Keep original key if not int-like
                return new_dict
            elif isinstance(obj, list):
                return [int_keys_if_possible(elem) for elem in obj]
            return obj
        # --- End Helper ---

        results = []
        total_execution_time = 0
        num_tests = 0

        Infinity = float('inf')
        NaN = float('nan')

        # The test_cases string from json.dumps will have string keys for graphs.
        raw_test_cases = [{"input": {"graph": {"0": {"1": 4, "7": 8}, "1": {"0": 4, "2": 8, "7": 11}, "2": {"1": 8, "3": 7, "8": 2, "5": 4}, "3": {"2": 7, "4": 9, "5": 14}, "4": {"3": 9, "5": 10}, "5": {"2": 4, "3": 14, "4": 10, "6": 2}, "6": {"5": 2, "7": 1, "8": 6}, "7": {"0": 8, "1": 11, "6": 1}, "8": {"2": 2, "6": 6, "7": 7}}, "source_node": 0}, "output": {"0": 0, "1": 4, "2": 12, "3": 19, "4": 21, "5": 11, "6": 9, "7": 8, "8": 14}}, {"input": {"graph": {"0": {"1": 10}, "1": {"0": 10}, "2": {"3": 5}, "3": {"2": 5}}, "source_node": 0}, "output": {"0": 0, "1": 10, "2": Infinity, "3": Infinity}}, {"input": {"graph": {"0": {}}, "source_node": 0}, "output": {"0": 0}}] 
        function_to_test_name = "dijkstra"
        
        # Make sure the function_to_test is available in the global scope
        if function_to_test_name not in globals():
            # Attempt to find it if it was defined inside a class (common for LLM output)
            # This is a simple heuristic and might need refinement.
            found_func = None
            for name, obj in list(globals().items()):
                if isinstance(obj, type):
                    if hasattr(obj, function_to_test_name):
                        method = getattr(obj, function_to_test_name)
                        if callable(method):
                            globals()[function_to_test_name] = method
                            found_func = True
                            break
            if not found_func:
                print(json.dumps({"error": f"Function '{function_to_test_name}' not found in the global scope or as a callable method of a defined class."}))
                sys.exit(1)
                
        function_to_test = globals()[function_to_test_name]
        
        for i, raw_test_case in enumerate(raw_test_cases):
            # Apply key conversion specifically to the 'graph' part of the input if it exists
            input_args = raw_test_case.get("input")
            if isinstance(input_args, dict) and 'graph' in input_args and isinstance(input_args['graph'], dict):
                # This assumes graph node IDs are the primary things needing int keys.
                # If other dicts in input_args need int keys, this logic needs adjustment.
                input_args['graph'] = int_keys_if_possible(input_args['graph'])
                # Also, if the source_node itself was stringified by an outer layer (unlikely here), it would need int()
                if 'source_node' in input_args and isinstance(input_args['source_node'], str):
                    try:
                        input_args['source_node'] = int(input_args['source_node'])
                    except ValueError:
                        pass # Keep as string if not int-like
            
            start_time = time.perf_counter()
            try:
                if isinstance(input_args, list):
                    actual_output = function_to_test(*input_args)
                elif isinstance(input_args, dict):
                    actual_output = function_to_test(**input_args)
                elif input_args is None:
                     actual_output = function_to_test()
                else:
                    actual_output = function_to_test(input_args)
                    
                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000
                total_execution_time += execution_time_ms
                num_tests += 1
                results.append({"test_case_id": i, "output": actual_output, "runtime_ms": execution_time_ms, "status": "success"})
            except Exception as e:
                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000
                error_output = {
                    "test_case_id": i,
                    "error": str(e), 
                    "error_type": type(e).__name__,
                    "runtime_ms": execution_time_ms,
                    "status": "error"
                }
                try:
                    json.dumps(error_output)
                except TypeError:
                    error_output["error"] = "Unserializable error object"
                results.append(error_output)
        
        final_output = {"test_outputs": results}
        if num_tests > 0:
            final_output["average_runtime_ms"] = total_execution_time / num_tests
        
        def custom_json_serializer(obj):
            if isinstance(obj, float):
                if obj == float('inf'):
                    return 'Infinity'
                elif obj == float('-inf'):
                    return '-Infinity'
                elif obj != obj:
                    return 'NaN'
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        
        print(json.dumps(final_output, default=custom_json_serializer))
        