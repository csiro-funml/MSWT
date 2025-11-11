#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=2gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=1        # Increased CPUs for DataLoader workers (H100 can handle more)
#SBATCH --output=slurm-%j.out     # Explicit output file (job ID will be inserted)
#SBATCH --error=slurm-%j.err      # Explicit error file

# Script to install ffmpeg on Linux server
# Supports multiple Linux distributions

echo "Detecting Linux distribution..."

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "Cannot detect distribution. Trying common methods..."
    DISTRO="unknown"
fi

echo "Detected distribution: $DISTRO"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if ffmpeg is already installed
if command_exists ffmpeg; then
    echo "ffmpeg is already installed:"
    ffmpeg -version | head -n 1
    exit 0
fi

# Installation based on distribution
case $DISTRO in
    ubuntu|debian)
        echo "Installing ffmpeg for Ubuntu/Debian..."
        if command_exists sudo; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        else
            echo "sudo not available. Trying without sudo..."
            apt-get update
            apt-get install -y ffmpeg
        fi
        ;;
    
    centos|rhel|fedora|rocky|almalinux)
        echo "Installing ffmpeg for CentOS/RHEL/Fedora..."
        if command_exists dnf; then
            if command_exists sudo; then
                sudo dnf install -y ffmpeg
            else
                dnf install -y ffmpeg
            fi
        elif command_exists yum; then
            if command_exists sudo; then
                sudo yum install -y epel-release
                sudo yum install -y ffmpeg
            else
                yum install -y epel-release
                yum install -y ffmpeg
            fi
        fi
        ;;
    
    arch|manjaro)
        echo "Installing ffmpeg for Arch Linux..."
        if command_exists sudo; then
            sudo pacman -S --noconfirm ffmpeg
        else
            pacman -S --noconfirm ffmpeg
        fi
        ;;
    
    *)
        echo "Unknown distribution. Trying common package managers..."
        
        # Try apt (Debian/Ubuntu)
        if command_exists apt-get; then
            echo "Trying apt-get..."
            if command_exists sudo; then
                sudo apt-get update && sudo apt-get install -y ffmpeg
            else
                apt-get update && apt-get install -y ffmpeg
            fi
        # Try yum (RHEL/CentOS)
        elif command_exists yum; then
            echo "Trying yum..."
            if command_exists sudo; then
                sudo yum install -y epel-release && sudo yum install -y ffmpeg
            else
                yum install -y epel-release && yum install -y ffmpeg
            fi
        # Try dnf (Fedora/newer RHEL)
        elif command_exists dnf; then
            echo "Trying dnf..."
            if command_exists sudo; then
                sudo dnf install -y ffmpeg
            else
                dnf install -y ffmpeg
            fi
        else
            echo "ERROR: Could not detect package manager."
            echo "Please install ffmpeg manually for your distribution."
            exit 1
        fi
        ;;
esac

# Verify installation
if command_exists ffmpeg; then
    echo ""
    echo "✓ ffmpeg installed successfully!"
    ffmpeg -version | head -n 1
else
    echo ""
    echo "✗ Installation may have failed. Please check the output above."
    echo "You may need to install ffmpeg manually or contact your system administrator."
    exit 1
fi

