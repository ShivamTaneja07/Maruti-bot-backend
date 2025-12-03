# mcp_logger.py

import os
from datetime import datetime, timezone
import pandas as pd

class ConversationLogger:
    def __init__(self, filepath: str = "conversation_log.xlsx"):
        self.filepath = filepath
        self.columns = [
            "session_id",
            "language",
            "session_start_timestamp",
            "turn_timestamp",
            "user_question",
            "bot_answer"
        ]

    def log_interaction(self, session_id: str, question: str, answer: str, language: str):
        """
        Logs a user question and bot answer to an Excel file.
        Each interaction is a new row.
        """
        try:
            # 1. Load existing data or create a new DataFrame
            if os.path.exists(self.filepath):
                df = pd.read_excel(self.filepath, engine='openpyxl')
            else:
                df = pd.DataFrame(columns=self.columns)

            # 2. Check if this is a new session to set the start timestamp
            session_entries = df[df['session_id'] == session_id]
            if session_entries.empty:
                # First time we see this session_id, so this is the start
                session_start_time = datetime.now(timezone.utc).isoformat()
            else:
                # Session already exists, reuse its original start time
                session_start_time = session_entries.iloc[0]['session_start_timestamp']

            # 3. Create the new row as a dictionary
            new_row = {
                "session_id": session_id,
                "language": language,
                "session_start_timestamp": session_start_time,
                "turn_timestamp": datetime.now(timezone.utc).isoformat(),
                "user_question": question,
                "bot_answer": answer
            }

            # 4. Append the new row and save the file
            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([df, new_row_df], ignore_index=True)
            
            # Use openpyxl engine to write to .xlsx format
            updated_df.to_excel(self.filepath, index=False, engine='openpyxl')
            
            print(f"Logged interaction for session: {session_id} to {self.filepath}")

        except PermissionError:
            print(f"🔥🔥🔥 PERMISSION DENIED: Could not write to {self.filepath}. Is the file open in Excel?")
        except Exception as e:
            print(f"🔥🔥🔥 An unexpected error occurred while writing to Excel log: {e}")