import os
import time
import subprocess
import itertools
import glob

# Parameter grid matching optimizer_full.py
param_grid = {
    "MOMENTUM_EMA_SPAN": [2, 3, 5],
    "MOMENTUM_THRESHOLD": [0.10, 0.15, 0.20],
    "STALE_HOURS": [4, 6, 8],
    "STALE_MOVE_THRESHOLD": [0.005, 0.01, 0.02],
    "INFORMED_RECENCY": [0.5, 0.65, 0.8]
}
keys = list(param_grid.keys())
combinations = list(itertools.product(*(param_grid[k] for k in keys)))
total_combinations = len(combinations)

def find_latest_log_path():
    base_dir = os.path.expanduser("~/.gemini/antigravity-cli/brain")
    log_files = []
    for root, dirs, files in os.walk(base_dir):
        if ".system_generated" in root and "tasks" in root:
            for file in files:
                if file.startswith("task-") and file.endswith(".log"):
                    full_path = os.path.join(root, file)
                    log_files.append((full_path, os.path.getmtime(full_path)))
    if not log_files:
        return None
    log_files.sort(key=lambda x: x[1], reverse=True)
    return log_files[0][0]

def format_time(seconds):
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"

try:
    while True:
        log_path = find_latest_log_path()
        if not log_path or not os.path.exists(log_path):
            print(f"\033[H\033[J{C_RED}No task log file found. Waiting...{C_RESET}")
            time.sleep(1)
            continue
            
        log_filename = os.path.basename(log_path)
        log_size_kb = os.path.getsize(log_path) / 1024
        
        with open(log_path, "r") as f:
            content = f.read()
            
        warnings_count = content.count("UserWarning")
        completed = max(0, min((warnings_count - 192) // 192 if warnings_count >= 192 else 0, total_combinations))
        pct = (completed / total_combinations) * 100
        
        # Get process stats
        pid = "N/A"
        cpu_usage = "N/A"
        mem_usage = "N/A"
        elapsed_seconds = None
        try:
            pid_out = subprocess.check_output(["pgrep", "-f", "optimizer_full.py"]).decode().strip()
            if pid_out:
                pid = pid_out.split('\n')[0]
                
                # CPU / Mem
                stats_out = subprocess.check_output(["ps", "-o", "%cpu=,%mem=", "-p", pid]).decode().strip()
                if stats_out:
                    cpu_usage, mem_usage = stats_out.split()
                    cpu_usage = f"{cpu_usage}%"
                    mem_usage = f"{mem_usage}%"
                
                # Elapsed
                etime_str = subprocess.check_output(["ps", "-o", "etime=", "-p", pid]).decode().strip()
                parts = etime_str.split('-')
                days = 0
                if len(parts) > 1:
                    days = int(parts[0])
                    time_part = parts[1]
                else:
                    time_part = parts[0]
                time_parts = list(map(int, time_part.split(':')))
                if len(time_parts) == 3:
                    h, m, s = time_parts
                else:
                    h = 0
                    m, s = time_parts
                elapsed_seconds = days * 86400 + h * 3600 + m * 60 + s
        except Exception:
            pass
            
        # Calculate rates and times
        avg_time = None
        remaining_seconds = None
        eta_str = "N/A"
        iter_per_min = 0.0
        if elapsed_seconds is not None and completed > 0:
            avg_time = elapsed_seconds / completed
            remaining_seconds = (total_combinations - completed) * avg_time
            eta_time = time.localtime(time.time() + remaining_seconds)
            eta_str = time.strftime("%I:%M:%S %p", eta_time)
            iter_per_min = 60.0 / avg_time
            
        # Current combination parameters
        current_params_str = "N/A"
        if completed < total_combinations:
            current_vals = combinations[completed]
            current_params = dict(zip(keys, current_vals))
            params_formatted = []
            for k, v in current_params.items():
                short_name = k.replace("MOMENTUM_", "MOM_").replace("THRESHOLD", "THRESH").replace("STALE_", "STALE_")
                params_formatted.append(f"{C_CYAN}{short_name}{C_RESET}={C_YELLOW}{v}{C_RESET}")
            current_params_str = ", ".join(params_formatted)
        else:
            current_params_str = f"{C_GREEN}COMPLETED!{C_RESET}"
            
        # Progress Bar Styling (25 blocks)
        filled_slots = int(pct // 4)
        bar = f"{C_GREEN}{'█' * filled_slots}{C_DIM}{'░' * (25 - filled_slots)}{C_RESET}"
        
        # Draw nice UI
        print("\033[H\033[J", end="")
        print(f"{C_BLUE}┌──────────────────────────────────────────────────────────┐{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}{C_GREEN}⚡ WEATHER BOT GRID SEARCH OPTIMIZER ⚡{C_RESET}                 {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}├──────────────────────────────────────────────────────────┤{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}Active Task:{C_RESET} {C_YELLOW}{log_filename:<12}{C_RESET}  │ {C_BOLD}PID:{C_RESET} {C_YELLOW}{pid:<8}{C_RESET}                   {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}├──────────────────────────────────────────────────────────┤{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}Progress:{C_RESET} {bar} {C_BOLD}{C_GREEN}{pct:.1f}%{C_RESET} ({completed}/{total_combinations})     {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}├──────────────────────────────────────────────────────────┤{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}CPU:{C_RESET} {C_MAGENTA}{cpu_usage:<7}{C_RESET} │ {C_BOLD}MEM:{C_RESET} {C_MAGENTA}{mem_usage:<7}{C_RESET} │ {C_BOLD}Warnings:{C_RESET}  {C_YELLOW}{warnings_count:<8}{C_RESET}   {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}├──────────────────────────────────────────────────────────┤{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}Elapsed Time:{C_RESET}      {C_CYAN}{format_time(elapsed_seconds):<12}{C_RESET} │ {C_BOLD}Log Size:{C_RESET}  {C_YELLOW}{log_size_kb:.1f} KB{C_RESET}      {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}Est. Remaining:{C_RESET}    {C_CYAN}{format_time(remaining_seconds):<12}{C_RESET} │ {C_BOLD}Speed:{C_RESET}     {C_YELLOW}{iter_per_min:.1f} iter/m{C_RESET} {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}│{C_RESET} {C_BOLD}Est. Completion:{C_RESET}   {C_CYAN}{eta_str:<12}{C_RESET} │                         {C_BLUE}│{C_RESET}")
        print(f"{C_BLUE}└──────────────────────────────────────────────────────────┘{C_RESET}")
        print(f"\n{C_BOLD}Current Configuration (Combination {completed + 1}/{total_combinations}):{C_RESET}")
        print(f"  {current_params_str}")
        print(f"\n{C_DIM}Press Ctrl+C to exit monitoring.{C_RESET}")
        
        time.sleep(1)
except KeyboardInterrupt:
    print(f"\n{C_GREEN}Exiting monitor.{C_RESET}")
