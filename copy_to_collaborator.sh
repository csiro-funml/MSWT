#!/bin/bash
# Script to efficiently copy prediction files to collaborator directory
# Usage:
#   ./copy_to_collaborator.sh <source_file> <destination_path>
#   
# Examples:
#   ./copy_to_collaborator.sh test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/test_data_prediction_long_spectral_reg.npz
#   ./copy_to_collaborator.sh /scratch3/wan410/operator_learning_model/path/to/file.npz /datasets/work/oa-tcch/work/forMichael/file.npz

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if path is remote (contains @ or :/)
is_remote() {
    [[ "$1" == *"@"* ]] || [[ "$1" == *":/"* ]]
}

# Function to copy file efficiently
copy_file() {
    local src="$1"
    local dest="$2"
    
    if [ ! -f "$src" ]; then
        echo -e "${YELLOW}Warning: Source file not found: $src${NC}"
        return 1
    fi
    
    # Create destination directory if it doesn't exist
    local dest_dir=$(dirname "$dest")
    if is_remote "$dest_dir"; then
        # For remote, use ssh to create directory
        local host_path=$(echo "$dest_dir" | sed 's/.*@\([^:]*\):\(.*\)/\1:\2/' | sed 's/\([^:]*\):\(.*\)/\2/')
        ssh $(echo "$dest_dir" | sed 's/\(.*@[^:]*\):.*/\1/') "mkdir -p $host_path" 2>/dev/null || true
    else
        mkdir -p "$dest_dir"
    fi
    
    # Determine copy method
    if is_remote "$dest"; then
        # Remote copy: use rsync (more efficient than scp)
        echo -e "${GREEN}Copying $src to remote: $dest${NC}"
        rsync -avh --progress "$src" "$dest"
    else
        # Local copy: use cp (fastest)
        echo -e "${GREEN}Copying $src to local: $dest${NC}"
        cp -v "$src" "$dest"
    fi
    
    # Verify copy
    if is_remote "$dest"; then
        # For remote, check file size
        local src_size=$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src" 2>/dev/null)
        echo "Source size: $src_size bytes"
    else
        if [ -f "$dest" ]; then
            local src_size=$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src" 2>/dev/null)
            local dest_size=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null)
            if [ "$src_size" -eq "$dest_size" ]; then
                echo -e "${GREEN}✓ Copy verified: $src_size bytes${NC}"
            else
                echo -e "${YELLOW}Warning: Size mismatch! Source: $src_size, Dest: $dest_size${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Error: Destination file not found after copy!${NC}"
            return 1
        fi
    fi
}

# Main execution
main() {
    # Check arguments
    if [ $# -lt 2 ]; then
        echo "Error: Missing arguments"
        echo ""
        echo "Usage:"
        echo "  ./copy_to_collaborator.sh <source_file> <destination_path>"
        echo ""
        echo "Examples:"
        echo "  ./copy_to_collaborator.sh test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/test_data_prediction_long_spectral_reg.npz"
        echo "  ./copy_to_collaborator.sh /scratch3/wan410/operator_learning_model/path/to/file.npz /datasets/work/oa-tcch/work/forMichael/file.npz"
        exit 1
    fi
    
    SOURCE_FILE="$1"
    DEST_FILE="$2"
    
    # If destination is a directory, append source filename
    if [ -d "$DEST_FILE" ] 2>/dev/null; then
        FILENAME=$(basename "$SOURCE_FILE")
        DEST_FILE="$DEST_FILE/$FILENAME"
    fi
    
    copy_file "$SOURCE_FILE" "$DEST_FILE"
    
    echo -e "${GREEN}✓ Copy completed${NC}"
}

# Run main function
main "$@"

