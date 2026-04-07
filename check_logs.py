import subprocess, re, sys

def validate_logs():
    print("Running inference.py to validate stdout format...")
    process = subprocess.Popen([sys.executable, "inference.py"], stdout=subprocess.PIPE, text=True)
    
    starts, steps, ends = 0, 0, 0
    start_pattern = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
    step_pattern = re.compile(r"^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=\S+$")
    end_pattern = re.compile(r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(?:-?\d+\.\d{2},?)+$")
    
    for line in process.stdout:
        line = line.strip()
        print(f"Captured: {line}")
        if line.startswith("[START]"): starts += bool(start_pattern.match(line))
        elif line.startswith("[STEP]"): steps += bool(step_pattern.match(line))
        elif line.startswith("[END]"): ends += bool(end_pattern.match(line))

    process.wait()
    if starts == 0 or steps == 0 or ends == 0:
        print("\n❌ FAILED: Regex mismatch. Check logs.")
        sys.exit(1)
    else:
        print(f"\n✅ PASSED: Found {starts} Starts, {steps} Steps, {ends} Ends matching regex exactly.")

if __name__ == "__main__":
    validate_logs()