import subprocess

def get_open_tabs():
    """Fetches all open tabs from Google Chrome."""
    script = """
    tell application "Google Chrome"
        set tab_data to ""
        set window_index to 1
        repeat with w in windows
            set tab_index to 1
            repeat with t in tabs of w
                set tab_data to tab_data & window_index & "," & tab_index & "," & title of t & "\n"
                set tab_index to tab_index + 1
            end repeat
            set window_index to window_index + 1
        end repeat
        return tab_data
    end tell
    """
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    return [tab for tab in result.stdout.strip().split('\n') if tab]

def _close_tab_by_id(window_index, tab_index):
    """Internal function to physically close the tab via AppleScript."""
    script = f"""
    tell application "Google Chrome"
        tell window {window_index}
            close tab {tab_index}
        end tell
    end tell
    """
    subprocess.run(['osascript', '-e', script])

def find_and_close_tab(target_name):
    """Searches open tabs for a keyword and closes the first match."""
    tabs = get_open_tabs()
    target_lower = target_name.lower().strip()
    
    for tab_info in tabs:
        # Split the string "Window,Tab,Title" into exactly 3 parts
        parts = tab_info.split(',', 2) 
        if len(parts) == 3:
            w_idx, t_idx, title = parts
            
            # Check if "youtube" is anywhere in the tab's title
            if target_lower in title.lower():
                print(f"[Jarvis System]: Found '{target_name}' at Window {w_idx}, Tab {t_idx}. Executing close protocol.")
                _close_tab_by_id(w_idx, t_idx)
                return True # Successfully closed
                
    print(f"[Jarvis System]: Target '{target_name}' not found on screen.")
    return False # Failed to find