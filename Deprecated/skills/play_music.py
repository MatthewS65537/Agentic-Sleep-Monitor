import os
import subprocess
import time

def play_music(music_file_path : str = "./music/1HRRussianWaltz.m4a") -> str:
    # Set Volume to make sure nothing BAD happens during sleep...
    # Most of the time, 4 clicks should be enough (25%)
    os.system("osascript -e 'set volume output volume 25'")
    # This value can be tuned later, or we can process all music to have all this volume

    # Make sure we have turned off the previous music. Easiest way is to close the music player
    os.system("osascript -e 'tell application \"QuickTime Player\" to quit'")

    time.sleep(2) # Give the system some time to register

    # Ensure the file path is in the correct format
    music_file_path = os.path.abspath(music_file_path)
    
    if not os.path.exists(music_file_path):
        print(f"Error: File not found - {music_file_path}")
        return False

    # Construct the AppleScript command
    applescript = f'''tell application "QuickTime Player"
        open POSIX file "{music_file_path}"
        play the front document
    end tell
    '''
    # applescript = f'''tell application "QuickTime Player"
    #     open POSIX file "{music_file_path}"
    # end tell
    # '''
    
    # Construct the full command
    command = ['osascript', '-e', applescript]
    
    try:
        # Execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return 0, "Music playback started successfully."
    except subprocess.CalledProcessError as e:
        return 1, f"Error executing command: {e}\nError output: {e.stderr}"
    
def list_available_music(music_dir):
    files = os.listdir(music_dir)
    return [f for f in files if os.path.isfile(os.path.join(music_dir, f)) and (f.endswith(".m4a") or f.endswith(".mp4") or f.endswith("mp3"))]

# TESTING PURPOSES ONLY
if __name__=="__main__":
    # play_music()
    print(list_available_music("../music"))
    play_music("./music/Rach2ndPianoConcerto.mp4")