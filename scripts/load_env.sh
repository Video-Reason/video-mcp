#!/bin/bash
# Load environment variables from .env file for AWS CLI usage

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Export AWS credentials
export AWS_ACCESS_KEY_ID=$(grep -E '^AWS_ACCESS_KEY_ID=' .env | cut -d '=' -f2- | tr -d '"')
export AWS_SECRET_ACCESS_KEY=$(grep -E '^AWS_SECRET_ACCESS_KEY=' .env | cut -d '=' -f2- | tr -d '"')
export AWS_DEFAULT_REGION=$(grep -E '^AWS_DEFAULT_REGION=' .env | cut -d '=' -f2- | tr -d '"')

echo "✓ AWS credentials loaded from .env"
echo "  Region: $AWS_DEFAULT_REGION"
