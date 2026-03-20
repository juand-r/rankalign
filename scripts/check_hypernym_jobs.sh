#!/bin/bash
# Check for hypernym jobs running on mll-a40-1 through mll-a40-7
#
# RECOMMENDED: First run ssh-add to cache your passphrase:
#   ssh-add ~/.ssh/id_rsa
#   (enter passphrase once)
#   ./check_hypernym_jobs.sh
#
# OR with passphrase inline: SSH_PASS="yourpassphrase" ./check_hypernym_jobs.sh

OUTPUT_FILE="hypernym_jobs_status.txt"
> "$OUTPUT_FILE"  # Clear/create the file

# Check if ssh-agent has the key loaded
if ssh-add -l &>/dev/null; then
    echo "SSH key is loaded in ssh-agent. Proceeding..."
    USE_EXPECT=false
else
    echo "SSH key not loaded in ssh-agent."
    # Check if passphrase is provided via environment variable
    if [ -z "$SSH_PASS" ]; then
        echo "You can either:"
        echo "  1. Run 'ssh-add ~/.ssh/id_rsa' first (recommended)"
        echo "  2. Run with: SSH_PASS=\"yourpassphrase\" ./check_hypernym_jobs.sh"
        echo ""
        echo "Or enter your SSH key passphrase now (will be used for all servers):"
        read -s -p "Passphrase: " SSH_PASS
        echo ""
    fi
    USE_EXPECT=true
fi

echo "Checking hypernym jobs on mll-a40 servers..."
echo "Results will be saved to: $OUTPUT_FILE"
echo ""

for N in 1 2 3 4 5 6 7; do
    SERVER="mll-a40-${N}.cs.utexas.edu"
    echo "Checking $SERVER..."
    
    echo "=== mll-a40-${N} ===" >> "$OUTPUT_FILE"
    
    if [ "$USE_EXPECT" = true ]; then
        # Use expect to provide SSH key passphrase
        expect -c "
            log_user 0
            set timeout 15
            spawn ssh -o StrictHostKeyChecking=no $SERVER {ps aux | grep hypernym | grep -v grep}
            expect {
                \"*passphrase*\" { send \"$SSH_PASS\r\"; log_user 1; exp_continue }
                \"*assword*\" { send \"$SSH_PASS\r\"; log_user 1; exp_continue }
                eof
            }
        " >> "$OUTPUT_FILE" 2>/dev/null
    else
        # SSH key is in agent, just run directly
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$SERVER" "ps aux | grep hypernym | grep -v grep" >> "$OUTPUT_FILE" 2>&1
    fi
    
    # Add a blank line between servers
    echo "" >> "$OUTPUT_FILE"
done

echo ""
echo "Done! Results saved to $OUTPUT_FILE"
echo ""
echo "--- Results ---"
cat "$OUTPUT_FILE"
