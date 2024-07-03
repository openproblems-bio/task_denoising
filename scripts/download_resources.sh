#!/bin/bash

set -e

echo ">> Downloading resources"

common/sync_resources/sync_resources \
  --input "s3://openproblems-data/resources_test/common/" \
  --output "resources_test/common" \
  --delete

common/sync_resources/sync_resources \
  --input "s3://openproblems-data/resources_test/denoising/" \
  --output "resources_test/denoising" \
  --delete