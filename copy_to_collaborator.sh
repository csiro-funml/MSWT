#!/bin/bash
#SBATCH --time=00:20:00           # Increased time for longer training with larger batches

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=32        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

# Script to efficiently copy prediction files to collaborator directory
# Usage:
#   ./copy_to_collaborator.sh [source_file] [destination_dir]
#   ./copy_to_collaborator.sh  # Uses default paths

set -e  # Exit on error

# Default paths (modify as needed)
SOURCE_DIR="${SOURCE_DIR:-/scratch3/wan410/operator_learning_model}"
DEST_DIR="${DEST_DIR:-/datasets/work/oa-tcch/work/forMichael}"

# File patterns to copy (modify as needed)
FILES_TO_COPY=(
    "test_data_prediction_long.npz"
    "test_data_prediction_long.pth"
)

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
    # If arguments provided, use them
    if [ $# -ge 1 ]; then
        SOURCE_FILE="$1"
        if [ $# -ge 2 ]; then
            DEST_DIR="$2"
        fi
        
        # Extract filename from source
        FILENAME=$(basename "$SOURCE_FILE")
        DEST_FILE="$DEST_DIR/$FILENAME"
        
        copy_file "$SOURCE_FILE" "$DEST_FILE"
        exit $?
    fi
    
    # Otherwise, use default behavior: copy all files from SOURCE_DIR to DEST_DIR
    echo "Copying files from $SOURCE_DIR to $DEST_DIR"
    echo "Files to copy: ${FILES_TO_COPY[@]}"
    echo ""
    
    # Find all matching files in source directory (recursively)
    found_files=0
    for pattern in "${FILES_TO_COPY[@]}"; do
        # Search in subdirectories
        while IFS= read -r -d '' file; do
            found_files=$((found_files + 1))
            rel_path="${file#$SOURCE_DIR/}"
            dest_file="$DEST_DIR/$rel_path"
            
            echo "Found: $file"
            copy_file "$file" "$dest_file"
            echo ""
        done < <(find "$SOURCE_DIR" -name "$pattern" -type f -print0 2>/dev/null)
    done
    
    if [ $found_files -eq 0 ]; then
        echo -e "${YELLOW}No files found matching patterns: ${FILES_TO_COPY[@]}${NC}"
        echo "Searching in: $SOURCE_DIR"
        echo ""
        echo "Usage examples:"
        echo "  # Copy specific file:"
        echo "  ./copy_to_collaborator.sh /path/to/test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/"
        echo ""
        echo "  # Copy with custom destination filename:"
        echo "  ./copy_to_collaborator.sh /path/to/test_data_prediction_long.npz /datasets/work/oa-tcch/work/forMichael/test_data_prediction_long_spectral_reg.npz"
        echo ""
        echo "  # Use environment variables:"
        echo "  SOURCE_DIR=/my/source DEST_DIR=/my/dest ./copy_to_collaborator.sh"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Copied $found_files file(s)${NC}"
}

# Run main function
main "$@"

