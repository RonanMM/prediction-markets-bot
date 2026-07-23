import subprocess
import sys
from pathlib import Path

def check_git_status():
    repo_root = Path(__file__).resolve().parent.parent
    cmd = ["git", "status", "--porcelain"]
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Git status failed: {res.stderr}")
        sys.exit(1)
    
    lines = res.stdout.strip().split('\n') if res.stdout.strip() else []
    modified_codebase_files = []
    
    for line in lines:
        if not line.strip():
            continue
        status_code = line[:2]
        filepath = line[3:].strip()
        
        # Check if modified file is inside codebase directories (src/, scripts/, or root python/md files)
        p = Path(filepath)
        parts = p.parts
        
        # Exclude allowed working output dirs: scratch/, .agents/
        if parts[0] in ['scratch', '.agents']:
            continue
            
        modified_codebase_files.append((status_code, filepath))
        
    print("==================== CODEBASE INTEGRITY VERIFICATION ====================")
    print(f"Repository Root: {repo_root}")
    print(f"Total Git Status Entries: {len(lines)}")
    print(f"Modified Codebase Files (src/, scripts/, root): {len(modified_codebase_files)}")
    
    if len(modified_codebase_files) == 0:
        print("\n✅ VERIFICATION SUCCESSFUL: ZERO codebase files have been modified!")
        print("   All outputs were strictly contained in scratch/ and .agents/.")
    else:
        print("\n❌ VERIFICATION FAILED: Codebase files were modified!")
        for status, fp in modified_codebase_files:
            print(f"   [{status}] {fp}")
        sys.exit(1)
    print("=========================================================================")

if __name__ == "__main__":
    check_git_status()
