# import os
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import pandas as pd
# from fastapi import HTTPException
# from vertexai.generative_models import GenerativeModel,SafetySetting  # Replace with the actual library import
# import re
# import json
# import networkx as nx
# import matplotlib.pyplot as plt
# import ast 
# # Vertexai Access
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

# # Initialize global variables
# chat_session = None
# extracted_column_data = None

# def generate_files_svgcontent(user_prompt: str, excel_data=None):
#     """Generates text with chat history using the Google Gemini model."""
#     global chat_session

#     system_instruction = "List out return the column names in list which suits to generate a detail DAG(Directed Acyclic Graph)"

#     try:
#         if chat_session is None:
#             model = GenerativeModel(model_name="gemini-1.5-pro-002", system_instruction=system_instruction)
#             chat_session = model.start_chat()

#         generation_config = {
#             "temperature": 1,
#             "top_p": 0.95,
#             "top_k": 40,
#             "max_output_tokens": 8192,
#             "response_mime_type": "text/plain",
#         }

#         safety_settings = [
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#         ]


#         # Build prompt
#         prompt = f"""
#             You are a designer AI. Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG(Directed Acyclic Graph).\n
#             {f'Excel Data: {excel_data}' if excel_data else ''}\n
#             User's original request: {user_prompt}\n
#             **Note: Don't provided me the code only return colums in array list**
#         """
#         print("User prompt: %s", prompt)

#         # Send message to chat session
#         response = chat_session.send_message([prompt], generation_config=generation_config,safety_settings=safety_settings)
#         parsed_result = response.text.strip()

#         return parsed_result

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

# def extract_excel_columns(file_path):

#     """Extract and display column names from an Excel file."""
#     try:
#         # Load the Excel file
#         df = pd.read_excel(file_path)
#         columns = df.columns.tolist()
#         rows = df.head(5).columns.tolist()
#         print("Extracted Column Names:")
#         for idx, col in enumerate(columns, start=1):
#             print(f"{idx}. {col}")
#         return columns,rows

#     except Exception as e:
#         logging.error(f"Error while reading the Excel file: {e}")
#         raise HTTPException(status_code=500, detail=f"Error while reading the Excel file: {str(e)}")

# #parsed the SVG response
# def parse_json_result(input_text):
#     if "```text" in input_text:
#         input_text = input_text.replace("```text", "").replace("```", "")
#     elif "```" in input_text:
#         input_text = input_text.replace("```", "")
#     input_text = input_text.replace("\n\n", "").replace("\n", "")
#     input_text = input_text.replace("\\\"", "\"").replace("\\", "")
#     try:  
#         cleaned_text = input_text.strip()
#         return input_text
#     except json.JSONDecodeError as e:
#         return None





# if __name__ == "__main__":

#     file_path = r"C:\\Users\\canaparthi\\Downloads\\Inputs-20241209T170717Z-001\\Inputs\\Harness Supplier Data\\IWCs\\25D2UC-2-100_E.wire.xls"
#     try:
#         # Extract and display columns
#         columns = extract_excel_columns(file_path)
#         rows = extract_excel_columns(file_path)
#         print(columns)
#         print(rows)

#         user_prompt = "Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG(Directed Acyclic Graph)."
#         excel_data = columns ,rows

#         # Generate SVG content
#         result = generate_files_svgcontent(user_prompt, excel_data)
#         print("Generated Result:")
#         extracted_column_data=result
#         # print(type(extracted_column_data))
#         print(extracted_column_data)
   
#         # Join the characters to form the proper string
#         joined_data = ''.join(extracted_column_data)

#         # Clean the string by removing unwanted characters (backticks, newlines)
#         cleaned_data = joined_data.replace("`", "").replace("\n", "")

#         # Now safely evaluate the cleaned string as a list
#         formatted_data_1 = ast.literal_eval(cleaned_data)

#         # Print the formatted list
#         print(formatted_data_1)


#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


# import os
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import ast
# from fastapi import HTTPException
# from vertexai.generative_models import GenerativeModel, SafetySetting
# import json

# # Vertexai Access
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

# # Initialize global variables
# chat_session = None
# extracted_column_data = None

# def generate_files_svgcontent(user_prompt: str, excel_data=None):
#     """Generates text with chat history using the Google Gemini model."""
#     global chat_session

#     system_instruction = "List out return the column names in list which suits to generate a detail DAG(Directed Acyclic Graph)"

#     try:
#         if chat_session is None:
#             model = GenerativeModel(model_name="gemini-1.5-pro-002", system_instruction=system_instruction)
#             chat_session = model.start_chat()

#         generation_config = {
#             "temperature": 1,
#             "top_p": 0.95,
#             "top_k": 40,
#             "max_output_tokens": 8192,
#             "response_mime_type": "text/plain",
#         }

#         safety_settings = [
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#             SafetySetting(
#                 category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
#                 threshold=SafetySetting.HarmBlockThreshold.OFF
#             ),
#         ]

#         # Build prompt
#         prompt = f"""
#             You are a designer AI. Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG(Directed Acyclic Graph).\n
#             {f'Excel Data: {excel_data}' if excel_data else ''}\n
#             User's original request: {user_prompt}\n
#             **Note: Don't provided me the code only return colums in array list**
#         """
#         print("User prompt: %s", prompt)

#         # Send message to chat session
#         response = chat_session.send_message([prompt], generation_config=generation_config, safety_settings=safety_settings)
#         parsed_result = response.text.strip()

#         return parsed_result

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

# def extract_excel_columns(file_path):
#     """Extract and display column names from an Excel file."""
#     try:
#         # Load the Excel file
#         df = pd.read_excel(file_path)
#         columns = df.columns.tolist()
#         rows = df.head(3).values.tolist()  # Get the first row
#         print("Extracted Column Names:")
#         for idx, col in enumerate(columns, start=1):
#             print(f"{idx}. {col}")
#         return columns, rows

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error while reading the Excel file: {str(e)}")

# def parse_json_result(input_text):
#     if "```text" in input_text:
#         input_text = input_text.replace("```text", "").replace("```", "")
#     elif "```" in input_text:
#         input_text = input_text.replace("```", "")
#     input_text = input_text.replace("\n\n", "").replace("\n", "")
#     input_text = input_text.replace("\\\"", "\"").replace("\\", "")
#     try:  
#         cleaned_text = input_text.strip()
#         return input_text
#     except json.JSONDecodeError as e:
#         return None

# # Function to draw DAG for each row in the Excel file
# def draw_dag_from_excel(file_path, formatted_columns, Raw_data_folder_path, Excel_data_folder_path):
#     """Read an Excel file, generate DAGs for each row, and save graphs and data into folders."""
#     try:
#         # Read the Excel file
#         df = pd.read_excel(file_path)
#         columns = list(df.columns)  # Get column names
#         row_data = df.values.tolist()  # Convert data to a list of rows
#         # Ensure the column names in the formatted_columns exist in the Excel data columns
#         matched_columns = [col for col in formatted_columns if col in columns]
#         if not matched_columns:
#             raise HTTPException(status_code=400, detail="No matching columns found to generate DAG")

#         if not row_data:
#             print("No data available.")
#             return

#         all_rows_data = []  # List to store all rows data for a single CSV

#         # Process each row in the Excel data
#         for idx, row in enumerate(row_data):
#             # Create a directed graph for the current row
#             G = nx.DiGraph()

#             for i in range(len(matched_columns) - 1):
#                 col1 = matched_columns[i]
#                 col2 = matched_columns[i + 1]

#                 # Find indices for columns in the row data
#                 if col1 in columns and col2 in columns:
#                     col1_idx = columns.index(col1)
#                     col2_idx = columns.index(col2)

#                     # Add nodes and create edges for the current row
#                     G.add_node(row[col1_idx])
#                     G.add_node(row[col2_idx])
#                     G.add_edge(row[col1_idx], row[col2_idx])

#             # Draw the graph for the current row
#             plt.figure(figsize=(8, 8))
#             nx.draw(G, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold")
#             plt.title(f"DAG (Directed Acyclic Graph) for Row {idx + 1}", fontsize=16)

#             # Save the DAG graph image
#             graph_image_path = os.path.join(Excel_data_folder_path, f'dag_graph_row_{idx + 1}.png')
#             plt.savefig(graph_image_path)
#             plt.close()

#             # Collect row data for a single CSV
#             row_data_dict = {col: row[columns.index(col)] for col in matched_columns}
#             all_rows_data.append(row_data_dict)

#         # Save all rows data into a single CSV file
#         all_rows_df = pd.DataFrame(all_rows_data)
#         all_rows_csv_path = os.path.join(Raw_data_folder_path, 'all_rows_data.csv')
#         all_rows_df.to_csv(all_rows_csv_path, index=False)

#         print(f"All row DAGs and corresponding CSV data have been saved to the specified folders.")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating DAGs: {str(e)}")


# if __name__ == "__main__":
#     file_path = r"C:\\Users\\canaparthi\\Downloads\\Inputs-20241209T170717Z-001\\Inputs\\Harness Supplier Data\\IWCs\\25D2UC-2-100_E.wire.xls"

#     try:
#         # Extract and display columns and first row from Excel
#         columns, rows = extract_excel_columns(file_path)
#         print("Columns:", columns)
#         print("First Row:", rows[0])

#         # Assume the formatted data from the previous process
#         user_prompt = "Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG (Directed Acyclic Graph)."
#         excel_data = columns, rows

#         # Generate SVG content (dummy process)
#         result = generate_files_svgcontent(user_prompt, excel_data)
#         parsed_data=parse_json_result(result)
#         print("Generated Result:",parsed_data )

#         # Join the characters to form the proper string
#         joined_data = ''.join(parsed_data)
        
#         # Clean the string by removing unwanted characters (backticks, newlines)
#         cleaned_data = joined_data.replace("`", "").replace("\n", "")

#         # Now safely evaluate the cleaned string as a list
#         formatted_data_1 = ast.literal_eval(cleaned_data)

#         # Print the formatted list
#         print("Formatted Columns:", formatted_data_1)

#         # Call the function to draw DAG
#         Raw_data_folder_path = 'C:\\Users\\canaparthi\\Downloads\\Raw_Data'
#         Excel_data_folder_path = 'C:\\Users\\canaparthi\\Downloads\\Graph_Data'

#         # Ensure folders exist
#         os.makedirs(Raw_data_folder_path, exist_ok=True)
#         os.makedirs(Excel_data_folder_path, exist_ok=True)

#         draw_dag_from_excel(file_path, formatted_data_1, Raw_data_folder_path, Excel_data_folder_path)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")







import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from fastapi import HTTPException
from vertexai.generative_models import GenerativeModel, SafetySetting
import json

# Vertexai Access
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"serverKey.json"

# Initialize global variables
chat_session = None
extracted_column_data = None

def generate_files_svgcontent(user_prompt: str, excel_data=None):
    """Generates text with chat history using the Google Gemini model."""
    global chat_session

    system_instruction = "List out return the column names in list which suits to generate a detail DAG(Directed Acyclic Graph)"

    try:
        if chat_session is None:
            model = GenerativeModel(model_name="gemini-1.5-pro-002", system_instruction=system_instruction)
            chat_session = model.start_chat()

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]

        # Build prompt
        prompt = f"""
            You are a designer AI. Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG(Directed Acyclic Graph).\n
            {f'Excel Data: {excel_data}' if excel_data else ''}\n
            User's original request: {user_prompt}\n
            **Note: Don't provided me the code only return colums in array list**
            **sample_output:["From", "Node 1", "Node 2",  "To","Wire Number","Pin1","Pin2"]**
        """
        print("User prompt: %s", prompt)

        # Send message to chat session
        response = chat_session.send_message([prompt], generation_config=generation_config, safety_settings=safety_settings)
        parsed_result = response.text.strip()

        return parsed_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

def extract_excel_columns(file_path):
    """Extract and display column names from an Excel file."""
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        columns = df.columns.tolist()
        rows = df.head(3).values.tolist()  # Get the first row
        print("Extracted Column Names:")
        for idx, col in enumerate(columns, start=1):
            print(f"{idx}. {col}")
        return columns, rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while reading the Excel file: {str(e)}")

def parse_json_result(input_text):
    if "```text" in input_text:
        input_text = input_text.replace("```text", "").replace("```", "")
    elif "```" in input_text:
        input_text = input_text.replace("```", "")
    input_text = input_text.replace("\n\n", "").replace("\n", "")
    input_text = input_text.replace("\\\"", "\"").replace("\\", "")
    try:  
        cleaned_text = input_text.strip()
        return input_text
    except json.JSONDecodeError as e:
        return None

# Function to draw DAG for each row in the Excel file
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from fastapi import HTTPException

def draw_dag_from_excel(file_path, formatted_columns, Raw_data_folder_path, Excel_data_folder_path):
    """Read an Excel file, generate DAGs for each row, and save graphs and data into folders."""
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        columns = list(df.columns)  # Get column names
        row_data = df.values.tolist()  # Convert data to a list of rows

        # Ensure the column names in the formatted_columns exist in the Excel data columns
        matched_columns = [col for col in formatted_columns if col in columns]
        if not matched_columns:
            raise HTTPException(status_code=400, detail="No matching columns found to generate DAG")

        if not row_data:
            print("No data available.")
            return

        all_columns_data = []  # List to store data for all columns

        # Process each row in the Excel data
        for idx, row in enumerate(row_data):
            # Create a directed graph for the current row
            G = nx.DiGraph()

            for i in range(len(matched_columns) - 1):
                col1 = matched_columns[i]
                col2 = matched_columns[i + 1]

                # Find indices for columns in the row data
                if col1 in columns and col2 in columns:
                    col1_idx = columns.index(col1)
                    col2_idx = columns.index(col2)

                    # Add nodes and create edges for the current row
                    G.add_node(row[col1_idx])
                    G.add_node(row[col2_idx])
                    G.add_edge(row[col1_idx], row[col2_idx])

            # Draw the graph for the current row
            plt.figure(figsize=(8, 8))
            nx.draw(G, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold")
            plt.title(f"DAG (Directed Acyclic Graph) for Row {idx + 1}", fontsize=16)

            # Save the DAG graph image
            graph_image_path = os.path.join(Excel_data_folder_path, f'dag_graph_row_{idx + 1}.png')
            plt.savefig(graph_image_path)
            plt.close()

            # Collect data from the columns and append to the all_columns_data
            row_data_dict = {col: row[columns.index(col)] for col in matched_columns}
            all_columns_data.append(row_data_dict)

        # Create a DataFrame from the collected column data
        all_columns_df = pd.DataFrame(all_columns_data)

        # Save all columns data into a single CSV file
        all_columns_csv_path = os.path.join(Raw_data_folder_path, 'combined_column_data.csv')
        all_columns_df.to_csv(all_columns_csv_path, index=False)

        print(f"All row DAGs and the combined column data have been saved to the specified folders.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating DAGs: {str(e)}")

#Process a Single  Xecel file 
def process_excel_file(file_path, Raw_data_folder_path, Excel_data_folder_path):
    """Process a single Excel file."""
    try:
        # Extract columns and rows from the current Excel file
        columns, rows = extract_excel_columns(file_path)
        print("Columns:", columns)
        print("First Row:", rows[0])

        # Assume the formatted data from the previous process
        user_prompt = "Your role is to verify the provided column names, records and return only column names in a list that suits to generate a detail DAG (Directed Acyclic Graph)."
        excel_data = columns, rows

        # Generate SVG content (dummy process)
        result = generate_files_svgcontent(user_prompt, excel_data)
        parsed_data = parse_json_result(result)
        print("Generated Result:", parsed_data)

        # Join the characters to form the proper string
        joined_data = ''.join(parsed_data)

        # Clean the string by removing unwanted characters (backticks, newlines)
        cleaned_data = joined_data.replace("`", "").replace("\n", "")

        # Now safely evaluate the cleaned string as a list
        formatted_data_1 = ast.literal_eval(cleaned_data)

        # Print the formatted list
        print("Formatted Columns:", formatted_data_1)

        # Call the function to draw DAG
        os.makedirs(Raw_data_folder_path, exist_ok=True)
        os.makedirs(Excel_data_folder_path, exist_ok=True)

        draw_dag_from_excel(file_path, formatted_data_1, Raw_data_folder_path, Excel_data_folder_path)
        print(f"Finished processing file: {file_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")


def process_all_excel_files(excel_folder, Raw_data_folder_path, Excel_data_folder_path):
    """Process all Excel files in a folder one by one."""
    try:
        excel_files = [f for f in os.listdir(excel_folder) if f.endswith('.xls') or f.endswith('.xlsx')]
        if not excel_files:
            print("No Excel files found in the folder.")
            return
        
        for file_name in excel_files:
            file_path = os.path.join(excel_folder, file_name)
            print(f"Processing file: {file_name}")
            
            # Process each file individually
            process_excel_file(file_path, Raw_data_folder_path, Excel_data_folder_path)
            print(f"Finished processing file: {file_name}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel files: {str(e)}")


if __name__ == "__main__":
    excel_folder_path = r"C:\\Users\\canaparthi\\Downloads\\Inputs-20241209T170717Z-001\\Inputs\\Harness Supplier Data\\IWCs"

    # Define folder paths for saving output
    Raw_data_folder_path = 'C:\\Users\\canaparthi\\Downloads\\Raw_Data'
    Excel_data_folder_path = 'C:\\Users\\canaparthi\\Downloads\\Graph_Data'

    # Process all Excel files in the specified folder
    process_all_excel_files(excel_folder_path, Raw_data_folder_path, Excel_data_folder_path)

