import gradio as gr
import requests
import json
import os
import tempfile
from datetime import datetime
import pandas as pd

# --- Configuration ---
SERVER_URL = "http://localhost:8001"
DATA_FILE = "myvoice_data.json"

# --- Data Persistence Functions ---
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"phrases": [], "history": []}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"phrases": [], "history": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- API Functions ---
def get_voices():
    try:
        resp = requests.get(f"{SERVER_URL}/voices", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("voices", [])
    except Exception as e:
        print(f"Error connecting to server: {e}")
    return ["F1"]  # Fallback default

def generate_audio(text, voice, speed, progress=gr.Progress()):
    if not text or not text.strip():
        return None, "Please enter some text."
    
    progress(0.2, desc="Contacting server...")
    
    payload = {
        "input": text,
        "voice": voice,
        "speed": float(speed),
        "total_steps": 10,  # Using defaults from your server
        "max_chunk_length": 300
    }
    
    try:
        resp = requests.post(f"{SERVER_URL}/v1/audio/speech", json=payload, timeout=60)
        progress(0.8, desc="Processing audio...")
        
        if resp.status_code == 200:
            # Save to a temp file so Gradio can play it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(resp.content)
                return f.name, None
        else:
            error_detail = resp.json().get("detail", "Unknown error")
            return None, f"Server Error: {error_detail}"
            
    except Exception as e:
        return None, f"Connection failed: {str(e)}"

# --- Logic Functions ---
def add_to_history(text, voice, speed):
    data = load_data()
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text,
        "voice": voice,
        "speed": speed
    }
    data["history"].insert(0, new_entry) # Add to top
    # Limit history to last 100 items
    if len(data["history"]) > 100:
        data["history"] = data["history"][:100]
    save_data(data)
    return data["history"]

def save_phrase(text, category, data_state):
    if not text or not text.strip():
        return data_state, "Cannot save empty text."
    
    # Ensure list structure exists
    if not data_state:
        data_state = load_data()
        
    new_phrase = {
        "text": text.strip(),
        "category": category if category else "Uncategorized"
    }
    
    # Check for duplicates
    for p in data_state["phrases"]:
        if p["text"] == new_phrase["text"] and p["category"] == new_phrase["category"]:
            return data_state, "Phrase already exists."
            
    data_state["phrases"].insert(0, new_phrase)
    save_data(data_state)
    return data_state, "Phrase saved!"

def delete_items(indices, data, list_type):
    """Generic delete function for phrases or history"""
    if not indices or not data:
        return data
    
    # Indices come sorted by Gradio (usually), but we need to delete from largest index to smallest 
    # to avoid shifting issues, or just filter the list.
    # Gradio Dataframe select_event returns a dict with 'index' list of integers (row numbers in the view)
    # However, since we might be filtering/searching, row numbers != list indices.
    # Simplified approach: The dataframe displayed is the full list (filtered by search logic happens in display)
    # Wait, Gradio Dataframe value IS the data displayed. 
    # So if we filter, the dataframe only shows filtered rows. We cannot delete from main list easily using filtered rows only.
    # SOLUTION: We will not pass the filtered DF back. We will pass the FULL list to a JS component or just handle it differently.
    # Easier approach for this specific script: The 'Delete' button will receive the selected indices relative to the CURRENTLY DISPLAYED dataframe.
    # We need to reconstruct the full list minus the selected items.
    pass

# Refined Logic for Delete/Play that works with the Gradio DataFrame update pattern
def handle_selection_processing(evt: gr.SelectData, current_df):
    # This isn't used for the main flow but useful for debugging
    pass

def get_filtered_phrases(search_query, category_filter):
    data = load_data()
    df = pd.DataFrame(data["phrases"])
    if df.empty:
        return df
    
    if category_filter:
        df = df[df["category"] == category_filter]
    if search_query:
        df = df[df["text"].str.contains(search_query, case=False, na=False)]
    return df

def get_filtered_history(search_query):
    data = load_data()
    df = pd.DataFrame(data["history"])
    if df.empty:
        return df
    
    if search_query:
        # Search in text and maybe voice name
        df = df[df["text"].str.contains(search_query, case=False, na=False)]
    return df

def perform_delete(selected_rows, full_df):
    # selected_rows is a list of indices (integers) from the dataframe component
    if not selected_rows:
        return full_df
    
    # pandas drop by index
    df_modified = full_df.drop(selected_rows)
    return df_modified

def commit_phrases(df_phrases):
    # Save the current state of the dataframe back to JSON
    data = load_data()
    data["phrases"] = df_phrases.to_dict(orient="records")
    save_data(data)
    return df_phrases

def commit_history(df_history):
    data = load_data()
    data["history"] = df_history.to_dict(orient="records")
    save_data(data)
    return df_history

# --- Gradio App Setup ---
with gr.Blocks(title="MyVoice", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🗣️ MyVoice")
    
    # Load initial voices
    initial_voices = get_voices()
    
    # State to keep track of current data without reloading file constantly
    # Though we will reload on tab switch to be safe and simple.
    
    with gr.Row():
        with gr.Column(scale=2):
            voice_dropdown = gr.Dropdown(choices=initial_voices, value=initial_voices[0] if initial_voices else "F1", label="Voice")
            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.05, value=1.05, label="Speed")
        with gr.Column(scale=5):
            main_input = gr.Textbox(label="Text to Speak", placeholder="Type something here...", lines=3)
        with gr.Column(scale=1):
            speak_btn = gr.Button("Speak", variant="primary", size="lg")
            
    audio_output = gr.Audio(label="Audio", autoplay=True)
    status_msg = gr.Textbox(label="Status", interactive=False, visible=False)

    # --- Logic for Speaking ---
    def on_speak(text, voice, speed):
        # 1. Generate Audio
        audio_path, error = generate_audio(text, voice, speed)
        
        # 2. Update History (only if successful)
        if audio_path:
            add_to_history(text, voice, speed)
            return audio_path, gr.Textbox(visible=False, value="")
        else:
            # Return None for audio to keep previous or stop, show error
            return None, gr.Textbox(value=error, visible=True)

    speak_btn.click(
        on_speak,
        inputs=[main_input, voice_dropdown, speed_slider],
        outputs=[audio_output, status_msg]
    )

    gr.Markdown("---")

    with gr.Tabs():
        # --- Tab 1: Saved Phrases ---
        with gr.TabItem("💾 Quick Phrases"):
            with gr.Row():
                new_category_input = gr.Textbox(label="New Category Name", placeholder="e.g. Greetings, Shopping")
                save_current_btn = gr.Button("Save Current Text to Phrases")
            
            save_status = gr.Textbox(label="", interactive=False)
            
            with gr.Row():
                search_phrase = gr.Textbox(placeholder="🔍 Search phrases...", show_label=False, scale=4)
                filter_category = gr.Textbox(placeholder="Filter by Category...", show_label=False, scale=2)
            
            phrases_df = gr.Dataframe(
                headers=["text", "category"],
                datatype=["str", "str"],
                label="Saved Phrases",
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                play_phrase_btn = gr.Button("▶️ Play Selected")
                load_phrase_btn = gr.Button("📝 Load Selected to Main")
                del_phrase_btn = gr.Button("🗑️ Delete Selected")

            # Functions for Phrases Tab
            def refresh_phrases(search, cat_filter):
                return get_filtered_phrases(search, cat_filter)

            def on_save_phrase(text, cat):
                data = load_data()
                _, msg = save_phrase(text, cat, data)
                # Refresh list
                new_df = get_filtered_phrases("", "") 
                return new_df, msg

            def on_play_selected(evt: gr.SelectData):
                # evt.index is the row number, evt.value is the row data [text, category]
                return evt.value[0]

            def on_load_selected(evt: gr.SelectData):
                return evt.value[0]

            # Event wiring
            search_phrase.change(refresh_phrases, inputs=[search_phrase, filter_category], outputs=phrases_df)
            filter_category.change(refresh_phrases, inputs=[search_phrase, filter_category], outputs=phrases_df)
            
            save_current_btn.click(
                on_save_phrase, 
                inputs=[main_input, new_category_input], 
                outputs=[phrases_df, save_status]
            )

            # For Dataframe selection in Gradio 4.x, we use the SelectData event on the component itself
            # However, buttons need to trigger actions. 
            # A tricky part: How to know which row is selected when a button is clicked?
            # Workaround: We use a hidden State to store the currently selected text.
            
            selected_phrase_text = gr.State("")
            
            phrases_df.select(
                fn=lambda evt: evt.value[0], # return text from the selected row
                outputs=selected_phrase_text
            )

            def play_selected_phrase(text, voice, speed):
                if not text: return None
                return on_speak(text, voice, speed)
            
            play_phrase_btn.click(
                play_selected_phrase,
                inputs=[selected_phrase_text, voice_dropdown, speed_slider],
                outputs=[audio_output, status_msg]
            )
            
            load_phrase_btn.click(
                lambda x: x,
                inputs=[selected_phrase_text],
                outputs=[main_input]
            )

            def delete_selected_phrases(df, selected_indices):
                # df is the current dataframe, selected_indices are the rows clicked
                # Note: Gradio Dataframe select events can be tricky with multi-select.
                # The simplest way in a basic script is often clicking the row to 'highlight' 
                # then clicking a button. But `select` event fires on click.
                # Here we rely on the fact that we pass the whole DF back.
                # Actually, to delete, we need to know the indices.
                # Let's assume single selection for simplicity or that the user sees selection.
                # The `select` event only gives the last clicked row index.
                
                if selected_indices is None:
                    return df
                
                # The index passed by .select() is a tuple (row, col) or similar depending on version?
                # In Gradio 4, SelectData.index is a tuple (row_index, column_index).
                # Since we bind the button click, we don't have the index in the button click args easily 
                # unless we stored it in a state.
                
                pass 
            
            # Better approach for Delete using Gradio:
            # We will implement a logic where `phrases_df` output is updated.
            # Since we can't easily pass "selected rows" to a button click without complex state management,
            # Let's use a simpler approach: The Checkbox feature isn't native in basic Dataframe without editing.
            # Let's stick to: Click row -> it populates "Selected Text". To delete, we match text.
            
            # Re-visiting delete logic:
            def delete_by_text(text_to_delete, current_df):
                if not text_to_delete: return current_df
                return current_df[current_df["text"] != text_to_delete]

            del_phrase_btn.click(
                delete_by_text,
                inputs=[selected_phrase_text, phrases_df],
                outputs=[phrases_df]
            ).then(
                commit_phrases,
                inputs=[phrases_df],
                outputs=[]
            )


        # --- Tab 2: History ---
        with gr.TabItem("🕒 History"):
            search_history = gr.Textbox(placeholder="🔍 Search history...", show_label=False)
            
            history_df = gr.Dataframe(
                headers=["timestamp", "text", "voice", "speed"],
                datatype=["str", "str", "str", "str"],
                label="History Log",
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                play_history_btn = gr.Button("▶️ Play Selected")
                del_history_btn = gr.Button("🗑️ Delete Selected")

            selected_history_text = gr.State("")

            def refresh_history(search):
                return get_filtered_history(search)

            search_history.change(refresh_history, inputs=[search_history], outputs=history_df)

            history_df.select(
                fn=lambda evt: evt.value[1], # Return text (column 1)
                outputs=selected_history_text
            )

            play_history_btn.click(
                play_selected_phrase, # reuse logic
                inputs=[selected_history_text, voice_dropdown, speed_slider],
                outputs=[audio_output, status_msg]
            )
            
            # Re-use delete logic (matching text)
            del_history_btn.click(
                delete_by_text,
                inputs=[selected_history_text, history_df],
                outputs=[history_df]
            ).then(
                commit_history,
                inputs=[history_df],
                outputs=[]
            )

    # Initial load of dataframes
    app.load(
        refresh_phrases, 
        inputs=[gr.Textbox(value=""), gr.Textbox(value="")], 
        outputs=[phrases_df]
    )
    app.load(
        refresh_history, 
        inputs=[gr.Textbox(value="")], 
        outputs=[history_df]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)